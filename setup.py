from typing import List
from setuptools import setup, find_packages

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="airfare-estimation",
    version='0.0.0',
    author="Saurabh Mandhalkar",
    author_email="saurabhmandhalkar03@gmail.com",
    packages=find_packages(),
    install_requires = ["scikit-learn","pandas","numpy"]
)