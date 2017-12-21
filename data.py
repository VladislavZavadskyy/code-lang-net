import gc, regex, os
import numpy as np
import tensorflow as tf

try:
    from IPython import display
    ipython=True
except:
    ipython=False

BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

class Config:
    line_count = None
    path = "/media/drive3"

    def __repr__(self):
        string = "Config:"
        attrs = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
        names = [regex.sub("_"," ",a).capitalize() for a in attrs]
        for attr, s in zip(attrs, names):
            a = getattr(self, attr)
            if isinstance(a, dict):
                string += "\n  "+s+": Dictionary"
            elif a is None:
                string += "\n  "+s+": None"
            else:
                string += "\n  "+s+": "+str(a)
        return string

class SmallConfig(Config):
  init_scale = 0.1
  max_grad_norm = 5
  num_layers = 2
  hidden_size = 200
  keep_prob = 1.0
  num_steps = 128
  batch_size = 1
  ignore_first = 15

class MediumConfig(Config):
  init_scale = 0.05
  max_grad_norm = 5
  num_layers = 2
  hidden_size = 650
  keep_prob = 0.5
  num_steps = 256
  batch_size = 1
  ignore_first = 15

class LangInput:
    def __init__(self, config):
        self.config = config

        self.num_steps = config.num_steps
        self.path = config.path
        self.files = [file for file in os.listdir(self.path) if\
                os.path.isfile(os.path.join(self.path,file))]
        if hasattr(config, "exclude_files"):
            self.files = [file for file in self.files if file not in config.exclude_files]
        elif hasattr(config, "include_files"):
            self.files = [file for file in self.files if file in config.include_files]
        files = self.files
        self.paths = [os.path.join(self.path, file) for file in files]
        self.prep_paths = [os.path.join(os.path.join(self.path,
                        'preprocessed'), file) for file in files]
        self.langs = [".".join(file.split(".")[:-1]) for file in files]
#         self.line_count = config.line_count or count_lines()
        self.cursors = {}
        self.num_classes = len(self.files)
#         self.epoch_size = int(sum(self.line_count.values())/len(self.line_count))
        self.itolang = dict(enumerate(self.langs))
        self.langtoi = dict(zip(self.itolang.values(),
                              self.itolang.keys()))

        self.itochar = dict(enumerate('\t\n !"#$%&\'()*+,-./0123456789:;<=>?'
        '@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'))
        self.chartoi = dict(zip(self.itochar.values(), self.itochar.keys()))
        self.num_chars = len(self.itochar)+1
        self.itochar[self.num_chars-1] = "<UNK>"
        self.config.vocab_size = self.num_chars

    def get_char(self, index):
        return self.chartoi.get(index,self.num_chars-1)

    @property
    def input_data(self):
        return tf.placeholder(tf.int32,[len(self.langs),self.num_steps],
        "input_placeholder")

    @property
    def targets(self):
        lb = tf.range(0, len(self.langs), 1, tf.int32)
        stack = tf.stack([lb for _ in range(self.num_steps)])
        return stack

    @staticmethod
    def preprocess(text):
        return regex.sub(u'[^\p{Latin}\p{Z}\p{S}\p{N}\p{P}\n\t]', u'', text)

    def count_lines(self, preprocessed=True):
        line_count = {}
        files = self.prep_paths if preprocessed else self.paths
        for p in files:
            line_count[p] = sum(1 for line in open(p))
        return line_count

    def next(self, numpy=True, preprocessed=True):
        chars = {}
        paths = self.prep_paths if preprocessed else self.paths
        for path, lang in zip(paths, self.langs):
            start = self.cursors.get(path,0)
            chars[lang] = ""
            with open(path, 'r+') as f:
                f.seek(start)
                remainder = self.num_steps
                while True:
                    try:
                        chars[lang] += f.read(remainder)
                    except UnicodeDecodeError:
                        self.cursors[path] = start+1
                        continue
                    self.cursors[path] = start+len(chars[lang])
                    remainder = self.num_steps-len(chars[lang])
                    if remainder>0:
                        f.seek(0)
                        self.cursors[path] = 0
                    else: break
        if numpy:
            np_data = []
            for v in chars.values():
                np_data.append(self.to_numpy(v))
            return np.array(np_data)
        return chars
    
    def to_numpy(self, string):
        return np.array(list(map(self.get_char,string)))
    
    def to_string(self, array):
        as_list = array.tolist()
        return "".join(list(map(self.itochar.get, as_list)))
    
    def preprocess_files(self, start_with=0, verbose=True):
        for i, lang in list(enumerate(self.langs))[start_with:]:
            with open(self.paths[i], 'r') as f:
                with open(self.prep_paths[i], 'w') as wf:
                    for li, line in enumerate(f):
                        p_line = self.preprocess(line)
                        wf.write(p_line)
                        if li%100000==0 and verbose:
                            if ipython:
                                display.clear_output(True)
                            print("Now processing "+lang)
                            print("\tOriginal:")
                            print(line)
                            print("\tPreprocessed:")
                            print(p_line)
            gc.collect()
