#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error 1001_main_Revised_v22.tex
