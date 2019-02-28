import os

from config import d, g

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == "__main__":
    dd, gg = d(), g()
    module = __import__("algorithms.%s" % d.config.algo, fromlist=[d.config.algo])
    algo_class = getattr(module, d.config.algo)
    algo_class(dd, gg, eval=True)
