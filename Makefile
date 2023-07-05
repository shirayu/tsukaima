
SHELL=/bin/bash

all: lint_node lint_python

TARGET_DIRS:=./tsukaima

flake8:
	find $(TARGET_DIRS) | grep '\.py$$' | xargs flake8
black:
	find $(TARGET_DIRS) | grep '\.py$$' | xargs black --diff | python ./tests/check_null.py
isort:
	find $(TARGET_DIRS) | grep '\.py$$' | xargs isort --diff | python ./tests/check_null.py
	
yamllint:
	find . \( -name node_modules -o -name .venv \) -prune -o -type f -name '*.yml' -print \
		| xargs yamllint --no-warnings

lint_python: flake8 black isort


pyright:
	npx pyright

markdownlint:
	find . -type d \( -name node_modules -o -name .venv \) -prune -o -type f -name '*.md' -print \
	| xargs npx markdownlint --config ./.markdownlint.json

lint_node: markdownlint pyright


style:
	find $(TARGET_DIRS) | grep '\.py$$' | xargs black
	find $(TARGET_DIRS) | grep '\.py$$' | xargs isort
