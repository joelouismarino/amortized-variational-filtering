import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from distributions import DiagonalGaussian, PointEstimate

# todo: add inverse auto-regressive flow to GaussianVariable
# todo: get conv modules working


class Dense(nn.Module):

    """Fully-connected (dense) layer with optional batch normalization, non-linearity, weight normalization, and dropout."""

    def __init__(self, n_in, n_out, non_linearity=None, batch_norm=False, weight_norm=False, dropout=0., initialize='glorot_uniform'):
        super(Dense, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(n_out, momentum=0.99)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear, name='weight')

        init_gain = 1.

        if non_linearity is None or non_linearity == 'linear':
            self.non_linearity = None
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
            init_gain = init.calculate_gain('relu')
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'selu':
            self.non_linearity = nn.SELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
            init_gain = init.calculate_gain('tanh')
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        else:
            raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')

        self.dropout = None
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)

        if initialize == 'normal':
            init.normal(self.linear.weight)
        elif initialize == 'glorot_uniform':
            init.xavier_uniform(self.linear.weight, gain=init_gain)
        elif initialize == 'glorot_normal':
            init.xavier_normal(self.linear.weight, gain=init_gain)
        elif initialize == 'kaiming_uniform':
            init.kaiming_uniform(self.linear.weight)
        elif initialize == 'kaiming_normal':
            init.kaiming_normal(self.linear.weight)
        elif initialize == 'orthogonal':
            init.orthogonal(self.linear.weight, gain=init_gain)
        elif initialize == '':
            pass
        else:
            raise Exception('Parameter initialization ' + str(initialize) + ' not found.')

        if batch_norm:
            init.normal(self.bn.weight, 1, 0.02)
            init.constant(self.bn.bias, 0.)

        init.constant(self.linear.bias, 0.)

    def random_re_init(self, re_init_fraction):
        pass

    def forward(self, input):
        output = self.linear(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        if self.dropout:
            output = self.dropout(output)
        return output


class Conv(nn.Module):

    """Basic convolutional layer with optional batch normalization, non-linearity, weight normalization and dropout."""

    def __init__(self, n_in, filter_size, n_out, non_linearity=None, batch_norm=False, weight_norm=False, dropout=0., initialize='glorot_uniform'):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(n_in, n_out, filter_size, padding=int(np.ceil(filter_size/2)))
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(n_out)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')

        if non_linearity is None:
            self.non_linearity = None
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'selu':
            self.non_linearity = nn.SELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        else:
            raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')

        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

        if initialize == 'normal':
            init.normal(self.conv.weight)
        elif initialize == 'glorot_uniform':
            init.xavier_uniform(self.conv.weight)
        elif initialize == 'glorot_normal':
            init.xavier_normal(self.conv.weight)
        elif initialize == 'kaiming_uniform':
            init.kaiming_uniform(self.conv.weight)
        elif initialize == 'kaiming_normal':
            init.kaiming_normal(self.conv.weight)
        elif initialize == 'orthogonal':
            init.orthogonal(self.conv.weight)
        elif initialize == '':
            pass
        else:
            raise Exception('Parameter initialization ' + str(initialize) + ' not found.')

        if batch_norm:
            init.constant(self.bn.weight, 1.)
            init.constant(self.bn.bias, 0.)

        init.constant(self.conv.bias, 0.)

    def forward(self, input):
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        if self.dropout:
            output = self.dropout(output)
        return output


class DenseInverseAutoRegressive(nn.Module):

    def __init__(self, n):
        super(DenseInverseAutoRegressive, self).__init__()
        self.mean = Dense(n, n)
        self.std = Dense(n, n)

    def forward(self, input):
        return (input - self.mean(input)) / self.std(input)


class MultiLayerPerceptron(nn.Module):

    """Multi-layered perceptron."""

    def __init__(self, n_in, n_units, n_layers, non_linearity=None, connection_type='sequential', batch_norm=False, weight_norm=False, dropout=0.):

        super(MultiLayerPerceptron, self).__init__()
        assert connection_type in ['sequential', 'residual', 'highway', 'concat_input', 'concat'], 'Connection type not found.'
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        n_in_orig = n_in

        if self.connection_type in ['residual', 'highway']:
            self.initial_dense = Dense(n_in, n_units, batch_norm=batch_norm, weight_norm=weight_norm)

        output_size = 0

        for _ in range(n_layers):
            layer = Dense(n_in, n_units, non_linearity=non_linearity, batch_norm=batch_norm, weight_norm=weight_norm, dropout=dropout)
            self.layers.append(layer)

            if self.connection_type == 'highway':
                gate = Dense(n_in, n_units, non_linearity='sigmoid', batch_norm=batch_norm, weight_norm=weight_norm)
                self.gates.append(gate)
            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_units
            elif self.connection_type == 'concat_input':
                n_in = n_units + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_units

            output_size = n_in

        self.n_out = output_size

    def random_re_init(self, re_init_fraction):
        for layer in self.layers:
            layer.random_re_init(re_init_fraction)
        if self.connection_type == 'highway':
            for gate in self.gates:
                gate.random_re_init(re_init_fraction)

    def forward(self, input):

        input_orig = input.clone()

        for layer_num, layer in enumerate(self.layers):
            if self.connection_type == 'sequential':
                input = layer(input)
            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_dense(input) + layer(input)
                else:
                    input += layer(input)
            elif self.connection_type == 'highway':
                gate = self.gates[layer_num]
                if layer_num == 0:
                    input = gate(input) * self.initial_dense(input) + (1 - gate(input)) * layer(input)
                else:
                    input = gate(input) * input + (1 - gate(input)) * layer(input)
            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer(input)), dim=1)
            elif self.connection_type == 'concat':
                input = torch.cat((input, layer(input)), dim=1)

        return input


class MultiLayerConv(nn.Module):

    """Multi-layer convolutional network."""

    def __init__(self, n_in, n_filters, filter_size, n_layers, non_linearity=None, connection_type='sequential', batch_norm=False, weight_norm=False, dropout=0.):
        super(MultiLayerConv, self).__init__()
        assert connection_type in ['sequential', 'residual', 'highway', 'concat_input', 'concat'], 'Connection type not found.'
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        n_in_orig = n_in

        if self.connection_type in ['residual', 'highway']:
            self.initial_conv = Conv(n_in, filter_size, n_filters, batch_norm=batch_norm, weight_norm=weight_norm)

        for _ in range(n_layers):
            layer = Conv(n_in, n_units, non_linearity=non_linearity, batch_norm=batch_norm, weight_norm=weight_norm, dropout=dropout)
            self.layers.append(layer)

            if self.connection_type == 'highway':
                gate = Conv(n_in, filter_size, n_filters, non_linearity='sigmoid', batch_norm=batch_norm, weight_norm=weight_norm)
                self.gates.append(gate)

            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_filters
            elif self.connection_type == 'concat_input':
                n_in = n_filters + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_filters

    def forward(self, input):

        input_orig = input.clone()

        for layer_num, layer in enumerate(self.layers):
            if self.connection_type == 'sequential':
                input = layer(input)

            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_conv(input) + layer(input)
                else:
                    input += layer(input)

            elif self.connection_type == 'highway':
                gate = self.gates[layer_num]
                if layer_num == 0:
                    input = gate * self.initial_conv(input) + (1 - gate) * layer(input)
                else:
                    input = gate * input + (1 - gate) * layer(input)

            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer(input)), dim=1)

            elif self.connection_type == 'concat':
                input = torch.cat((input, layer(input)), dim=1)

        return input


class DenseVariable(object):

    def __init__(self, batch_size, n_variables, n_orders_motion, const_prior_var, n_input, posterior_form='gaussian', learn_prior=True, dynamic=False):

        # variable hyper-parameters
        self.batch_size = batch_size
        self.n_variables = n_variables
        self.n_orders_motion = n_orders_motion
        self.posterior_form = posterior_form
        self.prior_form = 'gaussian'
        self.learn_prior = learn_prior
        self.dynamic = dynamic

        # prior (decoding) functions
        if self.learn_prior:
            self.prior_mean = [Dense(n_input[1], self.n_variables) for _ in range(self.n_orders_motion)]
            self.prior_log_var = None
            if not const_prior_var:
                self.prior_log_var = [Dense(n_input[1], self.n_variables) for _ in range(self.n_orders_motion)]

        # posterior (encoding) functions
        self.posterior_mean = [Dense(n_input[0], self.n_variables) for _ in range(self.n_orders_motion)]
        self.posterior_mean_gate = [Dense(n_input[0], self.n_variables, 'sigmoid') for _ in range(self.n_orders_motion)]
        if self.posterior_form == 'gaussian':
            self.posterior_log_var = [Dense(n_input[0], self.n_variables) for _ in range(self.n_orders_motion)]
            self.posterior_log_var_gate = [Dense(n_input[0], self.n_variables, 'sigmoid') for _ in range(self.n_orders_motion)]

        # posterior and prior densities
        self.posterior = [self.init_dist(self.posterior_form) for _ in range(self.n_orders_motion + 1)]
        self.prior = [self.init_dist(self.prior_form) for _ in range(self.n_orders_motion)]
        if self.learn_prior and const_prior_var:
            for p in self.prior:
                p.log_var_trainable()

    def init_dist(self, form):
        if form == 'gaussian':
            return DiagonalGaussian(Variable(torch.zeros(self.batch_size, self.n_variables)),
                                    Variable(torch.zeros(self.batch_size, self.n_variables)))
        elif form == 'point_estimate':
            return PointEstimate(Variable(torch.zeros(self.batch_size, self.n_variables)))
        else:
            raise Exception('Variable form ' + str(form) + ' not found.')

    def encode(self, input):
        # encode, update the posterior parameters, return sample
        for motion_order in range(self.n_orders_motion):
            mean, mean_gate = self.posterior_mean[motion_order](input), self.posterior_mean_gate[motion_order](input)
            mean = self.posterior[motion_order].mean + mean_gate * mean + self.posterior[motion_order + 1].mean
            self.posterior[motion_order].mean = mean
            if self.posterior_form == 'gaussian':
                log_var, log_var_gate = self.posterior_log_var[motion_order](input), self.posterior_log_var_gate[motion_order](input)
                # todo: figure out what the line below should actually be...
                log_var = self.posterior[motion_order].log_var + log_var_gate * log_var
                self.posterior[motion_order].log_var = log_var
        return torch.cat([p.sample(resample=True) for p in self.posterior], dim=1)

    def decode(self, input, generate=False):
        # decode, update the prior parameters, return sample
        samples = []
        for motion_order in range(self.n_orders_motion):
            if self.learn_prior:
                if self.dynamic:
                    self.prior[motion_order].mean = self.posterior[motion_order].mean.detach() + self.prior_mean[motion_order](input)
                else:
                    self.prior_mean = self.prior_mean[motion_order](input)
                if self.prior_log_var is not None:
                    self.prior[motion_order].log_var = self.prior_log_var[motion_order](input)
        if generate:
            return torch.cat([p.sample(resample=True) for p in self.prior], dim=1)
        else:
            return torch.cat([p.sample(resample=True) for p in self.posterior], dim=1)

    def error(self):
        return torch.cat([posterior.sample() - prior.mean.detach()
                          for posterior, prior in zip(self.posterior, self.prior)], dim=1)

    def norm_error(self):
        return torch.cat([(posterior.sample() - prior.mean.detach()) / torch.exp(prior.log_var.detach() + 1e-7)
                          for posterior, prior in zip(self.posterior, self.prior)], dim=1)

    def kl_divergence(self):
        return self.posterior.log_prob(self.posterior.sample()) - self.prior.log_prob(self.posterior.sample())

    def reset(self, from_prior=True):
        self.reset_mean(from_prior)
        self.reset_log_var(from_prior)

    def reset_mean(self, from_prior=True):
        for motion_order in range(self.n_orders_motion):
            value = None
            if from_prior:
                value = self.prior[motion_order].mean.data
            self.posterior[mottion_order].reset_mean(value)

    def reset_log_var(self, from_prior=True):
        if self.posterior_form == 'gaussian':
            for motion_order in range(self.n_orders_motion):
                value = None
                if from_prior:
                    value = self.prior[motion_order].log_var.data
                self.posterior[motion_order].reset_log_var(value)

    def trainable_mean(self):
        for motion_order in range(self.n_orders_motion):
            self.posterior[motion_order].mean_trainable()

    def trainable_log_var(self):
        if self.posterior_form == 'gaussian':
            for motion_order in range(self.n_orders_motion):
                self.posterior[motion_order].log_var_trainable()

    def eval(self):
        if self.learn_prior:
            self.prior_mean.eval()
            if self.prior_log_var is not None:
                self.prior_log_var.eval()
        for motion_order in range(self.n_orders_motion):
            self.posterior_mean[motion_order].eval()
            self.posterior_mean_gate[motion_order].eval()
            if self.posterior_form == 'gaussian':
                self.posterior_log_var[motion_order].eval()
                self.posterior_log_var_gate[motion_order].eval()

    def train(self):
        if self.learn_prior:
            self.prior_mean.train()
            if self.prior_log_var is not None:
                self.prior_log_var.train()
        for motion_order in range(self.n_orders_motion):
            self.posterior_mean[motion_order].train()
            self.posterior_mean_gate[motion_order].train()
            if self.posterior_form == 'gaussian':
                self.posterior_log_var[motion_order].train()
                self.posterior_log_var_gate[motion_order].train()

    def cuda(self, device_id=0):
        # place all modules on the GPU
        if self.learn_prior:
            self.prior_mean.cuda(device_id)
            if self.prior_log_var is not None:
                self.prior_log_var.cuda(device_id)
        for motion_order in range(self.n_orders_motion):
            self.posterior_mean[motion_order].cuda(device_id)
            self.posterior_mean_gate[motion_order].cuda(device_id)
            if self.posterior_form == 'gaussian':
                self.posterior_log_var[motion_order].cuda(device_id)
                self.posterior_log_var_gate[motion_order].cuda(device_id)
            self.prior[motion_order].cuda(device_id)
            self.posterior[motion_order].cuda(device_id)

    def parameters(self):
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        # returns the encoder parameters
        encoder_params = []
        for motion_order in range(self.n_orders_motion):
            encoder_params.extend(list(self.posterior_mean[motion_order].parameters()))
            encoder_params.extend(list(self.posterior_mean_gate[motion_order].parameters()))
            if self.posterior_form == 'gaussian':
                encoder_params.extend(list(self.posterior_log_var[motion_order].parameters()))
                encoder_params.extend(list(self.posterior_log_var_gate[motion_order].parameters()))
        return encoder_params

    def decoder_parameters(self):
        # returns the decoder parameters
        decoder_params = []
        if self.learn_prior:
            decoder_params.extend(list(self.prior_mean.parameters()))
            if self.prior_log_var is not None:
                decoder_params.extend(list(self.prior_log_var.parameters()))
            else:
                decoder_params.append(self.prior.log_var)
        return decoder_params

    def state_parameters(self):
        # returns the posterior (state) parameters
        state_params = []
        for motion_order in range(self.n_orders_motion):
            state_params[motion_order].extend(list(self.posterior[motion_order].state_parameters()))
        return state_params


class ConvGaussianVariable(object):

    def __init__(self, batch_size, n_variable_channels, filter_size, const_prior_var, n_input, update_form, learn_prior=True):

        self.batch_size = batch_size
        self.n_variable_channels = n_variable_channels
        assert update_form in ['direct', 'highway'], 'Variable update form not found.'
        self.update_form = update_form
        self.learn_prior = learn_prior

        if self.learn_prior:
            self.prior_mean = Conv(n_input[1], filter_size, self.n_variable_channels)
            self.prior_log_var = None
            if not const_prior_var:
                self.prior_log_var = Conv(n_input[1], filter_size, self.n_variable_channels)
        self.posterior_mean = Conv(n_input[0], filter_size, self.n_variable_channels)
        self.posterior_log_var = Conv(n_input[0], filter_size, self.n_variable_channels)

        if self.update_form == 'highway':
            self.posterior_mean_gate = Conv(n_input[0], filter_size, self.n_variable_channels, 'sigmoid')
            self.posterior_log_var_gate = Conv(n_input[0], filter_size, self.n_variable_channels, 'sigmoid')

        self.posterior = DiagonalGaussian()
        self.prior = DiagonalGaussian()
        if self.learn_prior and const_prior_var:
            self.prior.log_var_trainable()

    def encode(self, input):
        # encode the mean and log variance, update, return sample
        mean, log_var = self.posterior_mean(input), self.posterior_log_var(input)
        if self.update_form == 'highway':
            mean_gate = self.posterior_mean_gate(input)
            log_var_gate = self.posterior_log_var_gate(input)
            mean = mean_gate * self.posterior.mean + (1 - mean_gate) * mean
            log_var = log_var_gate * self.posterior.log_var + (1 - log_var_gate) * log_var
        self.posterior.mean, self.posterior.log_var = mean, log_var
        return self.posterior.sample()

    def decode(self, input, generate=False):
        # decode the mean and log variance, update, return sample
        mean, log_var = self.prior_mean(input), self.prior_log_var(input)
        self.prior.mean, self.prior.log_var = mean, log_var
        sample = self.prior.sample() if generate else self.posterior.sample()
        return sample

    def error(self):
        return self.posterior.sample() - self.prior.mean

    def norm_error(self):
        return self.error() / (torch.exp(self.prior.log_var) + 1e-7)

    def KL_divergence(self):
        return self.posterior.log_prob(self.posterior.sample()) - self.prior.log_prob(self.posterior.sample())

    def reset(self):
        self.reset_mean()
        self.reset_log_var()

    def reset_mean(self):
        self.posterior.reset_mean()

    def reset_log_var(self):
        self.posterior.reset_log_var()

    def trainable_mean(self, trainable=True):
        self.posterior.mean_trainable(trainable)

    def trainable_log_var(self, trainable=True):
        self.posterior.log_var_trainable(trainable)

    def cuda(self, device_id=0):
        # place all modules on the GPU
        self.prior_mean.cuda(device_id)
        self.prior_log_var.cuda(device_id)
        self.posterior_mean.cuda(device_id)
        self.posterior_log_var.cuda(device_id)
        if self.update_form == 'highway':
            self.posterior_mean_gate.cuda(device_id)
            self.posterior_log_var_gate.cuda(device_id)
        self.prior.cuda(device_id)
        self.posterior.cuda(device_id)

    def parameters(self):
        pass

    def encoder_parameters(self):
        pass

    def decoder_parameters(self):
        pass

    def state_parameters(self):
        return self.posterior.state_parameters()


class DenseLatentLevel(object):

    def __init__(self, batch_size, encoder_arch, decoder_arch, n_latent, n_det, n_orders_motion, encoding_form,
                 posterior_form='gaussian', const_prior_var=True, learn_prior=True, dynamic=False):

        self.batch_size = batch_size
        self.n_latent = n_latent
        self.encoding_form = encoding_form
        self.n_orders_motion = n_orders_motion
        self.posterior_form = posterior_form
        self.const_prior_var = const_prior_var
        self.learn_prior = learn_prior
        self.dynamic = dynamic

        self.encoder = MultiLayerPerceptron(**encoder_arch)
        self.decoder = MultiLayerPerceptron(**decoder_arch)

        variable_input_sizes = (encoder_arch['n_units'], decoder_arch['n_units'])
        self.latent_variable = DenseVariable(self.batch_size, self.n_latent, self.n_orders_motion, self.const_prior_var,
                                             variable_input_sizes, self.posterior_form, self.learn_prior, dynamic=self.dynamic)

        self.deterministic_encoder = Dense(variable_input_sizes[0], n_det[0]) if n_det[0] > 0 else None
        self.deterministic_decoder = Dense(variable_input_sizes[1], n_det[1]) if n_det[1] > 0 else None

    def get_encoding(self, input, in_out):
        encoding = input if in_out == 'in' else None
        if 'posterior' in self.encoding_form and in_out == 'out':
            encoding = input
        if ('top_error' in self.encoding_form and in_out == 'in') or ('bottom_error' in self.encoding_form and in_out == 'out'):
            error = self.latent.error()
            encoding = error if encoding is None else torch.cat((encoding, error), 1)
        if ('top_norm_error' in self.encoding_form and in_out == 'in') or ('bottom_norm_error' in self.encoding_form and in_out == 'out'):
            norm_error = self.latent.norm_error()
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), 1)
        return encoding

    def encode(self, input):
        # encode the input, possibly with errors, concatenate any deterministic units
        encoded = self.encoder(self.get_encoding(input, 'in'))
        output = self.get_encoding(self.latent.encode(encoded), 'out')
        if self.deterministic_encoder:
            det = self.deterministic_encoder(encoded)
            output = torch.cat((det, output), 1)
        return output

    def decode(self, input, generate=False):
        # decode the input, sample the latent variable, concatenate any deterministic units
        decoded = self.decoder(input)
        sample = self.latent.decode(decoded, generate=generate)
        if self.deterministic_decoder:
            det = self.deterministic_decoder(decoded)
            sample = torch.cat((sample, det), 1)
        return sample

    def kl_divergence(self):
        return self.latent.kl_divergence()

    def reset(self, from_prior=True):
        self.latent.reset(from_prior)

    def trainable_state(self):
        self.latent.trainable_mean()
        self.latent.trainable_log_var()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.latent.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.latent.train()

    def cuda(self, device_id=0):
        # place all modules on the GPU
        self.encoder.cuda(device_id)
        self.decoder.cuda(device_id)
        self.latent.cuda(device_id)
        if self.deterministic_encoder:
            self.deterministic_encoder.cuda(device_id)
        if self.deterministic_decoder:
            self.deterministic_decoder.cuda(device_id)

    def parameters(self):
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        encoder_params = []
        encoder_params.extend(list(self.encoder.parameters()))
        if self.deterministic_encoder:
            encoder_params.extend(list(self.deterministic_encoder.parameters()))
        encoder_params.extend(list(self.latent.encoder_parameters()))
        return encoder_params

    def decoder_parameters(self):
        decoder_params = []
        decoder_params.extend(list(self.decoder.parameters()))
        if self.deterministic_decoder:
            decoder_params.extend(list(self.deterministic_decoder.parameters()))
        decoder_params.extend(list(self.latent.decoder_parameters()))
        return decoder_params

    def state_parameters(self):
        return self.latent.state_parameters()


class ConvLatentLevel(object):
    pass
