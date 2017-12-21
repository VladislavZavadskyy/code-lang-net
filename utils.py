import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
    
def export_model(saver, model, session, input_node_names, output_node_name, model_name):
    tf.train.write_graph(session.graph_def, 'out', '%s_graph.pbtxt'%model_name)
    saver.save(session, 'out/%s.chkp'%model_name)

    freeze_graph.freeze_graph(input_graph = 'out/%s_graph.pbtxt'%model_name, 
                              input_saver = None,
                              input_binary = False, 
                              input_checkpoint = 'out/%s.chkp'%model_name, 
                              output_node_names = output_node_name, 
                              restore_op_name = "save/restore_all", 
                              filename_tensor_name = "save/Const:0",
                              output_graph = 'out/frozen_%s.pb'%model_name, 
                              clear_devices = True, 
                              initializer_nodes = "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_%s.pb'%model_name, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.int32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_%s.pb'%model_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("graph saved!")
    
def hex(r, g, b): 
    clamp = lambda x: max(0, min(x, 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

def heatmap(text,temp):
    string = ""
    if isinstance(temp,list):
        temp = np.array(temp)
    temp = temp / np.linalg.norm(temp)
    for letter, t in zip(text, temp):
        r = int(255/2+t*(255/2))
        b = int(255/2-t*(255/2))
        bg = hex(r,65,b)
        fg = hex(255,255,255)
        if letter=='\t':
            letter="&nbsp;"*4
        if letter==' ':
            letter="&nbsp;"
        if letter=='\n':
            string+='</br>'
        else:
            string+='<span style="background-color: %s; color: %s; font-size: 20px">%s</span>'%(bg,fg,letter)
    return string