# psyche
utils for machine learning 


## Usage
add your jupyter notebook this snippet.
You can use all function.

```python
%load_ext autoreload
%autoreload 2
try:
    import sys
    sys.path.append('../psyche/')
    from ShouldBeImported import *
except Exception as ex:
    print(ex)
```
