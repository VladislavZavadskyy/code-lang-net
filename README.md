# code-lang-net
A small 2-layer LSTM, classifies code by language for living.

Currently, it's capable of distinguishing between 13 languages, among which are Assembly, C/C++, C#, Clojure, CSS, HTML, Java, JavaScript, PHP, Python, R and Ruby. 

It achieves about 33% accuracy after first 20 symbols, 41% after first 40 symbols, 49% after 80 and so on.
I know how impressive that sounds, but keep in mind that it's a plain 2-layer LSTM with 200 nodes per layer.

Trained my [lots-of-code dataset](https://www.kaggle.com/zavadskyy/lots-of-code)

P.S.: Heatmap looks best with [oceans16](https://github.com/dunovank/jupyter-themes#oceans16-syntax) jupyter theme.
