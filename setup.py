from setuptools import setup, find_packages

setup(name='ocr4all_helpers',
      version='0.1',
      description='Different python scripts used in the OCR4all workflow.',
      url='https://github.com/OCR4all/OCR4all_helper-scripts',
      author='Nico Balbach',
      author_email='nico.balbach@informatik.uni-wuerzburg.de',
      packages=find_packages(),
      license='GPL-v3.0',
      #entry_points={
      #      'console_scripts': [
      #            'deda=deda.console.deda:main',
      #            'deda_eval_edition=deda.console.edition_evaluate:main',
      #            'deda_image_align_eval=deda.console.image_align_evaluate:main',
      #      ],
      #},
      install_requires=open("requirements.txt").read().split(),
      zip_safe=False)
