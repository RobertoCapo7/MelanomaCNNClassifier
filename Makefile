SHELL = /bin/bash

git_configuration := "yes"
github_repo_init := "yes"
repository_url := "https://github.com/RobertoCapo7/MelanomaCNNClassifier"
author_name := "RobertoCapo7"
author_email :="roberto.capolongo@live.it"
data_dir := "no"
DVC_configuration := "yes"

ifeq ($(OS),Windows_NT)
        ACTIVATE := venv\Scripts\activate  
    else
        ACTIVATE := source venv/bin/activate
    endif

.ONESHELL:
virtualenv:
	python3 -m venv venv
	@echo "Virtual environment created successfully"

install:
	$(ACTIVATE) && python -m pip install -r requirements.txt

update:
	$(ACTIVATE) && pip install --upgrade -r requirements.txt

check_variables:
    ifeq ($(git_configuration),"")
	    @echo "git_configuration is not set"
	    exit 1
    endif
    ifeq ($(github_repo_init),"")
	    @echo "github_repo_init is not set"
	    exit 1
    endif
    ifeq ($(repository_url),"")
	    @echo "repository_url is not set"
	    exit 1
    endif
    ifeq ($(author_name),"")
	    @echo "author_name is not set"
	    exit 1
    endif
    ifeq ($(author_email),"")
	    @echo "author_email is not set"
	    exit 1
    endif
    ifeq ($(data_dir),"")
	    @echo "data_dir is not specified"
	    exit 1
    endif
    ifeq ($(DVC_configuration),"")
	    @echo "DVC_configuration is not set"
	    exit 1
    endif

initialize_git:
    ifeq ($(git_configuration),"yes")
	    git config --global user.name $(author_name)
	    git config --global user.email $(author_email)
    else
	    @echo "Skipping git configuration"
    endif

initialize_repo:
	git init
    ifeq ($(github_repo_init),"yes")
	    git remote add origin https://github.com/RobertoCapo7/MelanomaCNNClassifier
		git add .
	    git commit -m 'initializing from cookiecutter'
	    git branch -M main
	    git push -u origin main
        ifneq ($(data_dir),"no")
	        mv $(data_dir) MelanomaCNNClassifier/data/raw
        else 
	        @echo "Skipping data directory setup as data_dir is set to 'no'."
        endif
    else
	    @echo "Skipping github repo init"
    endif

initialize_dvc:
    ifeq ($(DVC_configuration),"yes")
	    $(ACTIVATE) && dvc init
    endif
    ifeq ($(DVC_configuration),"no")
	    @echo "Skipping initialization as DVC."
    endif

setup_versioning: check_variables initialize_git initialize_repo initialize_dvc

start_tracking:
	@echo "Start tracking MlFlow..."
	mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri src/models/mlruns
	mlflow ui

start_QA:
	@echo "Start check Quality Assurance..."
	ruff check .
	ruff format .
	pynblint notebooks
	bandit -r src
	mypy src
	pytest test/
