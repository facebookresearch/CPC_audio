from setuptools import setup

setup(
   name='foo',
   version='1.0',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.com',
   packages=['eval', 'criterion'],  #same as name
   install_requires=['bar', 'greek'], #external packages as dependencies
)
