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
	import os
	HOME = os.environ['HOME']
    sys.path.append(f'{HOME}/gimpei-dot-files/opt/psyche/')
    from ShouldBeImported import *
except Exception as ex:
    print(ex)
```
