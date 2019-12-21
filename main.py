from functions.Topo import Topo
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Sample data: 
    _URL = "https://gist.githubusercontent.com/anonymous/d8975f76f5bcde7bd455/raw/831239b213fc29462db68f33caad3f05c57c0eff/topoplot_sample_data.csv"

    df = pd.read_csv(_URL)
    Sig = df[['x','y', 'signal']].values

    T = Topo(Sig)
    T.plot()
    plt.show()
