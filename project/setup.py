"""Setup."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 20日 星期日 08:42:09 CST
# ***
# ************************************************************************************/
#

import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
     name='rife',  
     version='0.0.1',
     author='Dell Du',
     author_email="",
     description="rife",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url='https://github.com/delldu/VideoRIFE',
     packages=['rife'],
     package_data={'rife': ['weights/*.pth']},
     include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
 )
