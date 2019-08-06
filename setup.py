# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# version info
NAME = "wechat-qr-detection"
VERSION = "0.0.6"

# requirements
install_requires = []
with open('requirements.txt', "r") as fp:
    for line in fp:
        if len(line.strip()) > 2:
            install_requires.append(line.strip())


# setup config
setup(
    name=NAME,
    version=VERSION,
    description="wechat qrcode detection",
    long_description=open("README.rst", "r").read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
    ],
    install_requires=install_requires,
    author="BiLiangDevelop && frkhit",
    url="https://github.com/BiLiangDevelop/wechat-qr-detection",
    author_email="frkhit@gmail.com",
    license="MIT",
    packages=["wechat_qr_detection", ],
    package_data={
        "": ["LICENSE", "README.rst", "MANIFEST.in"],
        "wechat_qr_detection": ["logo.png", ]
    },
    include_package_data=True,
)

