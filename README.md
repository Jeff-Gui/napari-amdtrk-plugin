# napari-amdtrk

[![License MIT](https://img.shields.io/pypi/l/napari-amdtrk.svg?color=green)](https://github.com/Jeff-Gui/napari-amdtrk/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-amdtrk.svg?color=green)](https://pypi.org/project/napari-amdtrk)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-amdtrk.svg?color=green)](https://python.org)
[![tests](https://github.com/Jeff-Gui/napari-amdtrk/workflows/tests/badge.svg)](https://github.com/Jeff-Gui/napari-amdtrk/actions)
[![codecov](https://codecov.io/gh/Jeff-Gui/napari-amdtrk/branch/main/graph/badge.svg)](https://codecov.io/gh/Jeff-Gui/napari-amdtrk)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-amdtrk)](https://napari-hub.org/plugins/napari-amdtrk)

Amend segmentation and track within napari manually

### [:eyes: watch a demo video](https://drive.google.com/file/d/1edC9Hw7e4FgD5QXDuUb45dAwRFt-TXFS/preview)

----------------------------------

### Input data structure

The essential input files include:
- An intensity image (`tif`)
- An object mask (`tif`)
- An object table (`csv`)
- A config file named `config.yaml`

    Within the config file, there should be 

----------------------------------

### Sample data

Sample cell track videos have been published with [_pcnaDeep: a fast and robust single-cell tracking method using deep-learning mediated cell cycle profiling_](10.1093/bioinformatics/btac602).


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-amdtrk` via [pip]:

    pip install napari-amdtrk




## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-amdtrk" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
