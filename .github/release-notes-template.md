<!-- TODO: one-line summary of this release -->

## Highlights

<!-- TODO: user-facing changes, one bullet each -->

## Install

New install on Linux, macOS, or WSL2 (authenticate Git first):

```bash
git clone git@github.com:aurekaresearch/OpenDDE-Harness-beta.git
cd OpenDDE-Harness-beta
./install.sh
```

New install on native Windows, in PowerShell:

```powershell
git clone https://github.com/aurekaresearch/OpenDDE-Harness-beta.git
cd OpenDDE-Harness-beta
.\install.ps1
```

Open a new terminal, then run:

```bash
ddeharness onboard
```

## Upgrade

Already running OpenDDE Harness? Upgrade in place -- configuration, sessions, and memory
are preserved:

```bash
ddeharness upgrade
```

`ddeharness upgrade` installs the latest stable release, so it does not pick up a
pre-release; rerun the installer above for that. Editable source checkouts are
never overwritten -- pull the checkout and rerun its development setup. On
native Windows the upgrade finishes in an external helper; wait for its
completion message before running OpenDDE Harness again.

## Release Status

- Version: `__VERSION__`
- Tag: `__TAG__`
- Stability: <!-- TODO: e.g. public preview patch / public preview minor -->
- Assets: wheel and source distribution attached to this release

## Notes

- OpenDDE Harness is still pre-1.0; CLI surfaces, plugin contracts, and runtime internals may continue to evolve.
- PyPI publishing is not enabled yet; the supported public install path uses the GitHub Release wheel asset.
