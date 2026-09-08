#!/bin/sh
# OpenDDE Harness installer (macOS / Linux).
#
#   Private repository: authenticate Git, clone the repository, then run
#   ./install.sh from the checkout root.
#
# Goal: a clean machine ends up able to run `ddeharness` from any
# directory with no manual steps. The script is idempotent -- it detects what
# is already present and only fills the gaps:
#   1. uv            (Python toolchain + package manager)
#   2. Node.js >= 22 (TUI runtime; a source build installs it privately here,
#      a wheel install leaves that to the first `ddeharness tui` run)
#   3. ddeharness     (installed as a global uv tool)
#
# POSIX sh on purpose (runs under dash/ash, not just bash).
set -eu

# --- config ---------------------------------------------------------------
MIN_NODE_MAJOR=22
OPENDDE_HARNESS_HOME="${OPENDDE_HARNESS_HOME:-${HOME:?HOME is required, or set OPENDDE_HARNESS_HOME explicitly}/.opendde_harness}"
NODE_RUNTIME_DIR="$OPENDDE_HARNESS_HOME/runtime"
INSTALL_SOURCE_DIR=""
NODE_BIN=""
PREPARE_ASSETS=0
CLIENT_ONLY=0
ASSETS_ONLY=0
ASSETS_DIR="${OPENDDE_HARNESS_ASSETS_DIR:-}"
ASSET_CHECKPOINT=""
DOWNLOAD_WORKERS=2

parse_options() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --client-only) CLIENT_ONLY=1; shift ;;
      --assets-only) ASSETS_ONLY=1; PREPARE_ASSETS=1; shift ;;
      --assets-dir)
        [ "$#" -ge 2 ] && [ -n "$2" ] || die "--assets-dir needs a persistent directory"
        ASSETS_DIR="$2"; shift 2 ;;
      --checkpoint)
        [ "$#" -ge 2 ] || die "--checkpoint needs opendde.pt or opendde_abag.pt"
        case "$2" in opendde.pt|opendde_abag.pt) ASSET_CHECKPOINT="$2" ;; *) die "Unsupported checkpoint: $2" ;; esac
        shift 2 ;;
      --download-workers)
        [ "$#" -ge 2 ] || die "--download-workers needs 1, 2, 3 or 4"
        case "$2" in 1|2|3|4) DOWNLOAD_WORKERS="$2" ;; *) die "--download-workers must be between 1 and 4" ;; esac
        shift 2 ;;
      -h|--help)
        printf '%s\n' 'Usage: sh install.sh [--client-only | --assets-only] [--assets-dir DIR] [--checkpoint NAME] [--download-workers N]' \
          'Default: install only the client. Onboarding uses a prebuilt Docker image with bundled sources, SolubleMPNN and ESM2.' \
          '--client-only: compatibility alias for the default client-only installation.' \
          '--assets-only: prepare/resume model weights without reinstalling the client (Harness tool weights, plus OpenDDE data in local folding mode).' \
          '--assets-dir DIR: Harness tool weights root (OPENDDE_HARNESS_WEIGHTS_DIR); set OPENDDE_ROOT_DIR for the OpenDDE data root.' \
          '--checkpoint: opendde.pt or opendde_abag.pt for local folding; default is the configured checkpoint.' \
          '--download-workers: 1-4 (default 2); optional OpenDDE checkpoint first, remaining assets in parallel.' \
          'Docker images, GPU drivers and large local search databases are not installed.'
        exit 0 ;;
      *) die "Unknown option: $1 (use --help)" ;;
    esac
  done
  [ "$ASSETS_ONLY" -eq 0 ] || [ "$CLIENT_ONLY" -eq 0 ] || die "--client-only and --assets-only cannot be combined"
}

# --- pretty output ---------------------------------------------------------
info()  { printf '\033[1;34m>\033[0m %s\n' "$1"; }
ok()    { printf '\033[1;32m+\033[0m %s\n' "$1"; }
warn()  { printf '\033[1;33m!\033[0m %s\n' "$1" >&2; }
die()   { printf '\033[1;31mx\033[0m %s\n' "$1" >&2; exit 1; }
have()  { command -v "$1" >/dev/null 2>&1; }

# --- 0. platform detection -------------------------------------------------
detect_platform() {
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os" in
    Darwin) NODE_OS="darwin" ;;
    Linux)  NODE_OS="linux" ;;
    *) die "Unsupported OS: $os (only macOS / Linux; on Windows use install.ps1)" ;;
  esac
  case "$arch" in
    arm64|aarch64) NODE_ARCH="arm64" ;;
    x86_64|amd64)  NODE_ARCH="x64" ;;
    *) die "Unsupported architecture: $arch" ;;
  esac
}

# --- 1. ensure uv ----------------------------------------------------------
ensure_uv() {
  if have uv; then
    ok "uv already installed ($(uv --version))"
    return
  fi
  info "uv not found, installing..."
  uv_installer="$(mktemp)"
  curl --connect-timeout 15 --max-time 60 --retry 2 -fsSL https://astral.sh/uv/install.sh -o "$uv_installer" \
    || die "Could not download uv; check access to astral.sh."
  sh "$uv_installer"
  rm -f "$uv_installer"
  # uv installs to ~/.local/bin (or $XDG_BIN_HOME) -- make it visible for the
  # rest of this script even before the shell profile is re-sourced.
  export PATH="$HOME/.local/bin:$PATH"
  have uv || die "uv still unavailable after install; check PATH (expected in ~/.local/bin)"
  ok "uv installed"
}

# --- 2. ensure Node >= 22 --------------------------------------------------
# Returns 0 if a usable system node is found.
system_node_ok() {
  have node || return 1
  v="$(node --version 2>/dev/null | sed 's/^v//; s/\..*//')"
  [ -n "$v" ] && [ "$v" -ge "$MIN_NODE_MAJOR" ] 2>/dev/null
}

# The pinned Node.js release, read from the client so both installers and
# `ddeharness tui` (which provisions Node for wheel installs) agree on it.
pinned_node_version() {
  ver="$(sed -n 's/^NODE_VERSION = "\([0-9.]*\)".*/\1/p' "$INSTALL_SOURCE_DIR/opendde_harness/cli/node_runtime.py" | head -n1)"
  [ -n "$ver" ] || die "Pinned Node.js version not found in opendde_harness/cli/node_runtime.py"
  printf 'v%s' "$ver"
}

# Print the path to a OpenDDE Harness-provisioned private node binary (first match), or
# return non-zero if none. Iterating the glob avoids passing multiple words to
# `[ -x ... ]` (which errors) when several versioned dirs linger.
private_node_bin() {
  for n in "$NODE_RUNTIME_DIR"/node-v22*/bin/node; do
    [ -x "$n" ] || continue
    # Actually run it -- a half-extracted / corrupt binary is +x but won't run,
    # and must NOT be mistaken for a ready runtime (else we'd never re-download).
    "$n" --version >/dev/null 2>&1 && { printf '%s' "$n"; return 0; }
  done
  return 1
}

ensure_node() {
  if system_node_ok && { [ -z "$INSTALL_SOURCE_DIR" ] || have npm; }; then
    NODE_BIN="$(command -v node)"
    ok "Node.js already meets requirement ($(node --version))"
    return
  fi
  # Already provisioned privately by a previous run?
  if pn="$(private_node_bin)"; then
    NODE_BIN="$pn"
    ok "OpenDDE Harness private Node already present ($pn)"
    return
  fi

  info "Node.js >= $MIN_NODE_MAJOR not found; downloading a private runtime (does not touch the system)..."
  ver="$(pinned_node_version)"
  pkg="node-${ver}-${NODE_OS}-${NODE_ARCH}"
  url="https://nodejs.org/dist/${ver}/${pkg}.tar.gz"
  mkdir -p "$NODE_RUNTIME_DIR"
  tmp="$(mktemp -d)"
  info "  $url"
  curl --connect-timeout 15 --max-time 300 --retry 2 -fL --progress-bar "$url" -o "$tmp/node.tar.gz" \
    || die "Node download failed: $url"

  # Supply-chain integrity: verify the tarball against the official
  # SHASUMS256.txt before extracting/executing it. Node publishes this file
  # next to every release.
  if curl --connect-timeout 10 --max-time 30 --retry 2 -fsSL "https://nodejs.org/dist/${ver}/SHASUMS256.txt" -o "$tmp/SHASUMS256.txt"; then
    expected="$(awk -v f="${pkg}.tar.gz" '$2==f {print $1}' "$tmp/SHASUMS256.txt")"
    if [ -n "$expected" ]; then
      if have shasum; then
        actual="$(shasum -a 256 "$tmp/node.tar.gz" | awk '{print $1}')"
      elif have sha256sum; then
        actual="$(sha256sum "$tmp/node.tar.gz" | awk '{print $1}')"
      else
        die "shasum or sha256sum is required to verify the Node download."
      fi
      if [ -n "$actual" ] && [ "$actual" != "$expected" ]; then
        rm -rf "$tmp"
        die "Node checksum mismatch (expected $expected, got $actual)"
      fi
      [ -n "$actual" ] && ok "Node tarball SHA256 verified"
    else
      die "SHASUMS256.txt did not list ${pkg}.tar.gz; the runtime was not installed."
    fi
  else
    die "Could not fetch SHASUMS256.txt; the unverified runtime was not installed."
  fi

  tar -xzf "$tmp/node.tar.gz" -C "$NODE_RUNTIME_DIR"
  rm -rf "$tmp"
  [ -x "$NODE_RUNTIME_DIR/$pkg/bin/node" ] || die "Node executable not found after extraction"
  # Run it once now: catches a libc mismatch (e.g. glibc tarball on Alpine/musl)
  # at install time instead of letting `ddeharness tui` fail later on the user's box.
  "$NODE_RUNTIME_DIR/$pkg/bin/node" --version >/dev/null 2>&1 \
    || die "Downloaded Node cannot run on this machine (possible libc mismatch, e.g. Alpine/musl). Install Node >= ${MIN_NODE_MAJOR} via your system package manager."
  ok "Node private runtime ready: $NODE_RUNTIME_DIR/$pkg"
  NODE_BIN="$NODE_RUNTIME_DIR/$pkg/bin/node"
  # opendde's find_node() globs ~/.opendde_harness/runtime/node-*/bin/node automatically,
  # so no PATH change is needed for `ddeharness tui` to find it.
}

# --- 3. install opendde ------------------------------------------------------
resolve_install_source() {
  if [ -f "$0" ]; then
    candidate_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
    if [ -f "$candidate_dir/pyproject.toml" ] && grep -q '^name = "opendde-harness"' "$candidate_dir/pyproject.toml"; then
      INSTALL_SOURCE_DIR="$candidate_dir"
      return
    fi
  fi
  wheel_url="${OPENDDE_HARNESS_WHEEL_URL:-}"
  if [ -z "$wheel_url" ]; then
    info "Resolving the latest OpenDDE Harness release from GitHub..."
    wheel_url="$(curl --connect-timeout 10 --max-time 30 -fsSL "https://api.github.com/repos/aurekaresearch/OpenDDE-Harness-beta/releases/latest" 2>/dev/null \
      | grep -oE 'https://[^"]*/opendde_harness-[^"]*\.whl' | head -n1)"
  fi
  if [ -z "$wheel_url" ]; then
    warn "GitHub API returned no release wheel; checking the release page."
    tag="$(curl --connect-timeout 10 --max-time 30 -fsS -o /dev/null -w '%{redirect_url}' \
      "https://github.com/aurekaresearch/OpenDDE-Harness-beta/releases/latest")" || tag=""
    case "$tag" in
      https://github.com/aurekaresearch/OpenDDE-Harness-beta/releases/tag/v*) version="${tag##*/}" ;;
      *) version="" ;;
    esac
    version="${version#v}"
    case "$version" in
      *.*.*.*) version="" ;;
      *.*.*) ;;
      *) version="" ;;
    esac
    if [ -n "$version" ]; then
      v_rest="${version#*.}"
      for field in "${version%%.*}" "${v_rest%%.*}" "${v_rest#*.}"; do
        case "$field" in
          ""|*[!0-9]*|0[0-9]*) version="" ;;
        esac
      done
    fi
    if [ -n "$version" ]; then
      wheel_url="https://github.com/aurekaresearch/OpenDDE-Harness-beta/releases/download/v${version}/opendde_harness-${version}-py3-none-any.whl"
    fi
  fi
  [ -n "$wheel_url" ] || die "No accessible published release was found. A private repository or an unpublished release cannot be installed anonymously. Use the authorized source-install command in README.md or a release wheel supplied by your administrator. No runtime was installed."
}

install_opendde_harness() {
  if [ -n "$INSTALL_SOURCE_DIR" ]; then
    script_dir="$INSTALL_SOURCE_DIR"
    info "Building a standalone installation from source: $script_dir"
    node_dir="$(dirname "$NODE_BIN")"
    PATH="$node_dir:$PATH" command -v npm >/dev/null 2>&1 \
      || die "The selected Node runtime has no npm; a complete Node installation is required to build the TUI."
    info "Building the TUI bundle..."
    ( cd "$script_dir/ui-tui" && PATH="$node_dir:$PATH" npm ci && PATH="$node_dir:$PATH" npm run build )
    [ -s "$script_dir/ui-tui/dist/entry.js" ] || die "TUI build did not produce entry.js; installation was stopped."
    # Pin to the locked dependency set so an install matches what we test.
    constraints="$(mktemp)"
    uv export --directory "$script_dir" --frozen --no-hashes --no-emit-project -o "$constraints" >/dev/null
    uv tool install --force --python 3.12 -c "$constraints" "$script_dir"
  else
    # Derive the locked-constraints URL from the wheel URL (same release dir) so
    # the constraints always match the wheel being installed, including when
    # OPENDDE_HARNESS_WHEEL_URL pins an older wheel. Missing asset / download failure ->
    # install without pinning rather than fail.
    constraints_url="${OPENDDE_HARNESS_CONSTRAINTS_URL:-}"
    if [ -z "$constraints_url" ]; then
      case "$wheel_url" in
        *.whl) constraints_url="${wheel_url%/*}/opendde-harness-constraints.txt" ;;
      esac
    fi
    c_args=""
    if [ -n "$constraints_url" ]; then
      constraints="$(mktemp)"
      if curl -fsSL "$constraints_url" -o "$constraints" 2>/dev/null; then
        c_args="-c $constraints"
      else
        warn "Could not download locked constraints; installing without version pinning."
      fi
    fi
    info "  installing $wheel_url"
    # shellcheck disable=SC2086  # $c_args is an intentional word-split option pair.
    uv tool install --force --python 3.12 $c_args "$wheel_url"
  fi
  # Ensure ~/.local/bin (uv tool bin dir) is on PATH for future shells.
  uv tool update-shell || true
  tool_bin="$(uv tool dir --bin)"
  [ -x "$tool_bin/ddeharness" ] || die "Installation did not create ddeharness in $tool_bin. The selected package may predate this CLI entry point."
  "$tool_bin/ddeharness" --version
  if [ -n "$NODE_BIN" ]; then
    OPENDDE_HARNESS_NODE="$NODE_BIN" "$tool_bin/ddeharness" tui --check \
      || die "The installed TUI failed its startup check; installation is not complete."
  else
    # Wheel installs need no Node up front: the check installs a private Node.js
    # runtime when the system has none (set OPENDDE_HARNESS_NO_NODE_INSTALL=1 to forbid it).
    "$tool_bin/ddeharness" tui --check \
      || die "The installed TUI failed its startup check; installation is not complete."
  fi
  ok "ddeharness installed"
}

# --- main ------------------------------------------------------------------
main() {
  parse_options "$@"
  have curl || die "curl is required; please install it first"
  # Read before installing so the closing hint can tell a first run from an
  # upgrade; the install itself never writes config.json (the wizard does).
  if [ -f "$OPENDDE_HARNESS_HOME/config.json" ]; then
    had_config=1
  else
    had_config=0
  fi
  detect_platform
  if [ "$ASSETS_ONLY" -eq 0 ]; then
    resolve_install_source
  elif [ -f "$0" ]; then
    candidate_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
    if [ -f "$candidate_dir/pyproject.toml" ] && grep -q '^name = "opendde-harness"' "$candidate_dir/pyproject.toml"; then
      INSTALL_SOURCE_DIR="$candidate_dir"
    fi
  fi
  if [ "$PREPARE_ASSETS" -eq 1 ]; then
    [ -z "$ASSETS_DIR" ] || export OPENDDE_HARNESS_WEIGHTS_DIR="$ASSETS_DIR"
    info "Harness tool weights: ${OPENDDE_HARNESS_WEIGHTS_DIR:-$HOME/.cache/opendde-harness}; OpenDDE data (local folding only): ${OPENDDE_ROOT_DIR:-$HOME/.cache/opendde}."
  fi
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  ensure_uv
  if [ "$ASSETS_ONLY" -eq 0 ]; then
    # Building the TUI from source needs Node with npm before the client exists;
    # a wheel install lets `ddeharness tui --check` provision Node itself.
    [ -z "$INSTALL_SOURCE_DIR" ] || ensure_node
    install_opendde_harness
  fi
  if [ "$PREPARE_ASSETS" -eq 1 ]; then
    tool_bin="$(uv tool dir --bin)"
    [ -x "$tool_bin/ddeharness" ] || die "ddeharness is missing; run a full installation of this version first"
    (
      if [ -n "$INSTALL_SOURCE_DIR" ]; then
        export PYTHONPATH="$INSTALL_SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
      fi
      set -- compute prepare --assets-only --download-workers "$DOWNLOAD_WORKERS"
      [ -z "$ASSET_CHECKPOINT" ] || set -- "$@" --checkpoint "$ASSET_CHECKPOINT"
      "$tool_bin/ddeharness" "$@"
    ) || die "Client installation and compute preparation are separate: asset preparation is incomplete. Fix the error and rerun with --assets-only; partial downloads are retained."
  fi

  printf '\n'
  if [ "$ASSETS_ONLY" -eq 1 ]; then
    ok "Compute assets prepared. The installed client and application configuration were not changed."
    return
  fi
  if [ "$had_config" = 1 ]; then
    ok "OpenDDE Harness updated. Your config in $OPENDDE_HARNESS_HOME is unchanged."
    printf '\n    \033[1mddeharness\033[0m    # continue where you left off\n\n'
    if [ -n "$INSTALL_SOURCE_DIR" ]; then
      printf '  tip: rerun the README installation command to update this source build.\n\n'
    else
      printf '  tip: next time you can upgrade in place with \033[1mddeharness upgrade\033[0m\n\n'
    fi
  else
    ok "All set! Open a new terminal (or source your shell profile), then run:"
    printf '\n    \033[1mddeharness\033[0m    # sets you up on first run, then opens the TUI\n\n'
  fi
  if ! printf '%s' "$PATH" | grep -q "$HOME/.local/bin"; then
    warn "Your current PATH does not include ~/.local/bin yet -- open a new terminal, or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
  fi
}

main "$@"
