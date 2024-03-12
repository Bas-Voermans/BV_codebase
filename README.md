# BV_codebase
My codebase of often used machine-learning and bioinformatics stuff. Developed for personal use, but free to share for colleagues and friends alike.

## Machine learning pipelines
blablabla

## Installing a Conda Environment from a `.yml` File

To recreate a conda environment from a `.yml` file, you can use the `conda env create` command. Here are the steps:

1. **Open your terminal**: You can do this by searching for "terminal" on your computer.

2. **Navigate to the directory containing the `.yml` file**: Use the `cd` command followed by the path to the directory. For example, if your `.yml` file is in a folder named "environments" on your Desktop, you would type `cd Desktop/environments`.

3. **Create the environment**: Type `conda env create -f environment.yml` in your terminal, replacing "environment.yml" with the name of your `.yml` file. This will create a new conda environment with the same name and packages as specified in the `.yml` file.

4. **Activate the new environment**: Once the environment is created, you can activate it using `conda activate env_name`, replacing "env_name" with the name of your new environment.

Remember, you need to have Anaconda or Miniconda installed on your computer to use conda commands. If you haven't installed it yet, please follow the instructions on their official websites.