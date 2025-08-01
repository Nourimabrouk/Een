install:
	python -m pip install -e .[dev,ai]

pre-commit-install:
	pre-commit install --install-hooks

agent:
	spawn-agent $(path)

.PHONY: install pre-commit-install agent