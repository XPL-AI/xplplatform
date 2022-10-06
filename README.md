#xpl platform
### Setup development environment
1. Install [Python3.9](https://www.python.org/downloads)
2. Install basic Python libraries:
   ~~~~
   pip3 install virtualenv setuptools
   ~~~~
   
3. Run Install Certificates.command.:
   ~~~~
   On MAC OS:
   navigate to your Applications/Python 3.9/ folder and double click the "Install Certificates.command"
   ~~~~
4. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/quickstart)
5. Login to Google cloud. This will authorize access to all google cloud resources
   including: cloud storage, firestore, BigQuery, AppEngine etc.
   ~~~~
   gcloud auth login
   ~~~~
   ~~~~
   gcloud auth application-default login
   ~~~~
6. Install [git](https://git-scm.com/downloads)
7. Clone [*xplplatform*](https://source.cloud.google.com/) git repository
   ~~~~
   git clone https://source.cloud.google.com/xplplatform01/xplplatform
   ~~~~
8. Set environment variables:
   ~~~~
   export XPL_ENV=DEV
   export XPL_CODE_DIR=~/path_to_xplplatform_repository
   ~~~~
9. Register *xpl-init* command
   ~~~~
   sudo ln -s ~/path_to_xplplatform_repository/xpl-init.sh /usr/local/bin/xpl-init
   chmod +x ~/path_to_xplplatform_repository/xpl-init.sh
   ~~~~
10. Activate code area that you want to start working with.
   For example if you want to start working with xpl.coordinate you run:
   ~~~~
   source xpl-init xpl.coordinate
   ~~~~
   or
   ~~~~
   . xpl-init xpl.coordinate
   ~~~~
   Or you can use a shortcut that will cut xpl* prefix and start from the name of component:
   ~~~~
   . xpl-init coordinate
   ~~~~
   . xpl-init will create a virtual environment for specified project
   and install dependencies.
   
    **xpl-init**
   
    *-p project.name* a project to activate. follows the folder structure. examples: xpl.sandbox, xpl.data
    
    *-r* resets the virtual environment and reinstall dependencies from scratch
    
### Switch between  DEV and PROD environments
1. Initialize PROD environment
   When asked to give a name - type: prod
   When asked to select project - choose xpl-platform-prod
   ~~~~
   gcloud init
   ~~~~
2. Switch to prod configuration
   ~~~~
   gcloud config configurations activate prod
   ~~~~
3. Set XPL_ENV environment variable
   ~~~~
   export XPL_ENV=PROD
   ~~~~
   List local configurations
   ~~~~
   gcloud config configurations list
   ~~~~
   