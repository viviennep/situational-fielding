from model_scripts.double_catch_prob import catch_prob
import pickle as pkl, cloudpickle as cpkl

with open('models/normal-catch-prob.pkl','rb') as f:
    normal = pkl.load(f)
with open('models/wall-catch-prob.pkl','rb') as f:
    wall = pkl.load(f)
cp = catch_prob(normal,wall)
with open('models/catch-prob.pkl','wb') as f:
    cpkl.dump(cp,f)


