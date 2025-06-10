export VLLM_TARGET_DEVICE=empty 
pip install -r requirements/build.txt . -i https://pypi.antfin-inc.com/simple/
pip install build -i https://pypi.antfin-inc.com/simple/
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CPLUS_INCLUDE_PATH=/usr/lib/gcc/x86_64-openEuler-linux/12/../../../../include/c++/12:/usr/lib/gcc/x86_64-openEuler-linux/12/../../../../include/c++/12/x86_64-openEuler-linux:/usr/lib/gcc/x86_64-openEuler-linux/12/../../../../include/c++/12/backward:/usr/lib/gcc/x86_64-openEuler-linux/12/include:/usr/local/include:/usr/include
mkdir -p dist
python -m build --no-isolation .