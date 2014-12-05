import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import application
from blocks.graph import ComputationGraph
from blocks.parallel import Mixer
from blocks.attention import SequenceContentAttention
from blocks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks import Tanh, MLP, Linear, LinearMaxout, Identity
from blocks.initialization import Constant, IsotropicGaussian
from blocks.sequence_generators import (SequenceGenerator,
                                        AttentionTransition,
                                        LookupFeedback,
                                        LinearReadout,
                                        SoftmaxEmitter)


floatX = theano.config.floatX


INPUT_DIM = 3
RNN_DIM = 10
STATE_DIM = 12
MAXOUT_DIM = 14
LABEL_DIM = 5
N_STEPS = 10

rnn1 = GatedRecurrent(dim=RNN_DIM,
                      activation=Tanh(),
                      weights_init=IsotropicGaussian(0, .3),
                      use_update_gate=True,
                      use_reset_gate=True)

bidir = Bidirectional(prototype=rnn1, name='bidir')
x = T.tensor3('x')
mask = T.matrix('mask')
y = T.lmatrix('y')

in_w = Linear(input_dim=INPUT_DIM, output_dim=RNN_DIM,
              weights_init=IsotropicGaussian(0, .1), biases_init=Constant(0),
              name='in_w')
in_w.initialize()  # without this line everything will be zeros
h = in_w.apply(x)

embedding = bidir.apply(inps=h, update_inps=h, reset_inps=h, mask=mask,
                        iterate=True)

mlp = MLP(activations=[Tanh(name='layer0'), Identity(name='layer1')],
          dims=[2 * RNN_DIM + STATE_DIM, 10, 1],
          weights_init=IsotropicGaussian(.01),
          biases_init=Constant(0),
          name='mlp')
mlp.allocate()
mlp.initialize()


class GRNNTransition(GatedRecurrent):
    def __init__(self, attended_dim, **kwargs):
        super(GRNNTransition, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.linear = Linear(attended_dim, self.dim, biases_init=Constant(0),
                             weights_init=IsotropicGaussian(0.01))
        self.children.append(self.linear)

    @application(contexts=['attended', 'attended_mask'])
    def apply(self, *args, **kwargs):
        # remove the original context names?
        for context in GRNNTransition.apply.contexts:
            kwargs.pop(context)
        return super(GRNNTransition, self).apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        # TODO: what is the meaning of delegation? It seems to mean that the
        # context is not used in this case...
        return super(GRNNTransition, self).apply

    def get_dim(self, name):
        if name == 'attended':
            return self.attended_dim
        if name == 'attended_mask':
            return 0
        return super(GRNNTransition, self).get_dim(name)

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return self.linear.apply(kwargs['attended'][0])


transition = GRNNTransition(dim=STATE_DIM, name="transition",
                            attended_dim=RNN_DIM * 2)


attention_mech = SequenceContentAttention(state_names=transition.apply.states,
                                          match_dim=STATE_DIM,
                                          energy_computer=mlp,
                                          name='attention')
mixer = Mixer([name for name in transition.apply.sequences
               if name != 'mask'],
              attention_mech.take_look.outputs[0], name='mixer')
attention_trans = AttentionTransition(transition, attention_mech, mixer,
                                      attended_name='attended',
                                      attended_mask_name='attended_mask',
                                      name='attention_trans')

maxout = LinearMaxout(STATE_DIM, MAXOUT_DIM, num_pieces=2,
                      weights_init=IsotropicGaussian(0, .01),
                      biases_init=Constant(0))

readout = LinearReadout(readout_dim=LABEL_DIM,
                        source_names=['states'],
                        emitter=SoftmaxEmitter(name='emitter'),
                        feedbacker=LookupFeedback(LABEL_DIM, STATE_DIM,
                                                  name='feedback'),
                        weights_init=IsotropicGaussian(0.01),
                        biases_init=Constant(0),
                        name='readout')

generator = SequenceGenerator(readout=readout,
                              transition=transition,
                              attention=attention_mech,
                              weights_init=IsotropicGaussian(0.01),
                              biases_init=Constant(0),
                              name='generator')


costs = generator.cost(y, attended=embedding, attended_mask=mask)

cost_val = theano.function([y, x, mask], costs)
result = generator.generate(n_steps=N_STEPS, batch_size=1,
                            attended=embedding, attended_mask=mask)
graph = ComputationGraph(result[1])
updates = graph.updates
forward = theano.function([x, mask], result[1], updates=updates)

X = np.asarray(np.random.normal(0, 1, (10, 1, INPUT_DIM)), dtype=floatX)
Y = np.asarray(np.random.random_integers(0, 3, (10, 1)))
M = np.ones((10, 1), dtype=floatX)
M[-3:] = 0
print cost_val(Y, X, M)
print forward(X, M)
