#!/bin/bash
julia --project=. -e 'import Pkg; Pkg.update(); Pkg.instantiate(); using Pluto; Pluto.run()'
