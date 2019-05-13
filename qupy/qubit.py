from __future__ import division
import numpy as np
import math
import qupy


def _to_tuple(x):
    if np.issubdtype(type(x), np.integer):
        x = (x,)
    return x


class Qubits:
    """
    Creating qubits.

    Args:
        size (:class:`int`):
            Number of qubits.
        dtype:
            Data type of the data array.
        gpu (:class:`int`):
            GPU machine number.

    Attributes:
        data (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
            The state of qubits.
        size:
            Number of qubits.
        dtype:
            Data type of the data array.
    """

    def __init__(self, size, dtype=np.complex128, xp=np, **kargs):

        self.xp = xp
        self.size = size
        self.dtype = dtype
        self.basic_operator = qupy.Operator(xp=self.xp, dtype=dtype)

        self.state = self.xp.zeros([2] * self.size, dtype=dtype)
        self.state[tuple([0] * self.size)] = 1

    def set_state(self, state):
        """set_state(self, state)

        Set state.

        Args:
            state (:class:`str` or :class:`list` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                If you set state as :class:`str`, you can set state \\state>
                (e.g. state='0110' -> \\0110>.)
                otherwise, qubit state is set that you entered as state.
        """
        if isinstance(state, str):
            if len(state) != self.state.ndim:
                raise ValueError('There were {} qubits prepared, but you specified {} qubits.'
                                 .format(self.state.ndim, len(state)))
            self.state = self.xp.zeros_like(self.state)
            self.state[tuple([int(i) for i in state])] = 1
        else:
            self.state = self.xp.asarray(state, dtype=self.dtype)
            if self.state.ndim == 1:
                self.state = self.state.reshape([2] * self.size)

    def get_state(self, flatten=True):
        """get_state(self, flatten=True)

        Get state.

        Args:
            flatten (:class:`bool`):
                If you set flatten=False, you can get data format used in QuPy.
                otherwise, you get state reformated to 1D-array.
        """
        if flatten:
            return self.state.flatten()
        return self.state

    def gate(self, operator, target, control=None, control_0=None):
        """gate(self, operator, target, control=None, control_0=None)

        Gate method.

        Args:
            operator (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Unitary operator
            target (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operated qubits
            control (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operate target qubits where all control qubits are 1
            control_0 (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operate target qubits where all control qubits are 0
        """
        xp = self.xp

        target = _to_tuple(target)
        control = _to_tuple(control)
        control_0 = _to_tuple(control_0)

        operator = xp.asarray(operator, dtype=self.dtype)

        if operator.size != 2 ** (len(target) * 2):
            raise ValueError('You must set operator.size==2^(len(target)*2)')

        if operator.shape[0] != 2:
            operator = operator.reshape([2] * int(math.log2(operator.size)))

        c_slice = [slice(None)] * self.size
        if control is not None:
            for _c in control:
                c_slice[_c] = slice(1, 2)
        if control_0 is not None:
            for _c in control_0:
                c_slice[_c] = slice(0, 1)
        c_slice = tuple(c_slice)

        c_idx = list(range(self.size))
        t_idx = list(range(self.size))
        for i, _t in enumerate(target):
            t_idx[_t] = self.size + i
        o_idx = list(range(self.size, self.size + len(target))) + list(target)

        character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        o_index = ''.join([character[i] for i in o_idx])
        c_index = ''.join([character[i] for i in c_idx])
        t_index = ''.join([character[i] for i in t_idx])
        subscripts = '{},{}->{}'.format(o_index, c_index, t_index)
        self.state[c_slice] = xp.einsum(subscripts, operator, self.state[c_slice])

    def project(self, target):
        """projection(self, target)

        Projection method.

        Args:
            target (None or :class:`int` or :class:`tuple` of :class:`int`):
                projected qubits

        Returns:
            :class:`int`: O or 1.
        """
        xp = self.xp

        state = xp.split(self.state, [1], axis=target)
        p = [self._to_scalar(xp.sum(state[i] * xp.conj(state[i])).real) for i in (0, 1)]
        obs = np.random.choice([0, 1], p=p)

        if obs == 0:
            self.state = xp.concatenate((state[obs] / math.sqrt(p[obs]), xp.zeros_like(state[obs])), target)
        else:
            self.state = xp.concatenate((xp.zeros_like(state[obs]), state[obs] / math.sqrt(p[obs])), target)
        return obs

    def expect(self, observable, n_trial=-1, flip_rate=0):
        """expect(self, observable)

        Method to get expected value of observable.

        Args:
            observable (:class:`dict` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Physical quantity operator.
                If you input :class:`numpy.ndarray` or :class:`cupy.ndarray` as observable,
                this method returns :math:`\\langle \\psi | \\mathrm{observable} | \\psi>`,
                where :math:`\\psi>` is the states of qubits.
                If you use :class:`dict` input, you have to set
                {'operator1': coef1, 'operator2': coef2, 'operator3': coef3, ...},
                such as {'XIX': 0.32, 'YYZ': 0.11, 'III': 0.02}.
                If you input :class:`dict` as observable,
                this method returns :math:`\\sum_i coef_i \\langle \\psi | \\mathrm{operator}_i | \\psi>`.

        Returns:
            :class:`float`: Expected value.
        """
        xp = self.xp

        if isinstance(observable, dict):
            if (flip_rate < 0) or (flip_rate > 1):
                raise ValueError('You should set 0 <= flip_rate <= 1. Actual: {}.'.format(flip_rate))

            ret = 0
            org_state = self.state

            for key, value in observable.items():
                self.state = xp.copy(org_state)
                if len(key) != self.size:
                    raise ValueError('Each key length must be {}, but len({}) is {}.'.format(self.size, key, len(key)))

                for i, op in enumerate(key):
                    if op in 'XYZ':
                        self.gate(getattr(self.basic_operator, op), target=i)
                    elif op != 'I':
                        raise ValueError('Keys of input must not include {}.'.format(op))

                e_val = xp.dot(xp.conj(org_state.flatten()), self.state.flatten())  # i,i
                e_val = self._to_scalar(xp.real(e_val))

                if flip_rate > 0:
                    e_val = e_val * (1 - 2 * flip_rate)

                if n_trial > 0:
                    probability = (e_val + 1) / 2
                    probability = np.clip(probability, 0, 1)
                    e_val = (np.random.binomial(n_trial, probability) / n_trial) * 2 - 1

                ret += e_val * value

            self.state = org_state
            return ret

        else:
            if flip_rate != 0:
                raise ValueError('Sorry, flip_rate is supported only in the case that observable type is dictionary.')

            if observable.size != self.state.size * self.state.size:
                raise ValueError('operator.size must be {}. Actual: {}'.format(self.state.size ** 2, observable.size))

            observable = xp.asarray(observable, dtype=self.dtype)
            if observable.shape[0] != self.state.size:
                observable = observable.reshape((self.state.size, self.state.size))

            if n_trial <= 0:
                ret = xp.dot(xp.conj(self.state.flatten()), xp.dot(observable, self.state.flatten()))  # i,ij,j
                return self._to_scalar(xp.real(ret))
            else:
                w, v = xp.linalg.eigh(observable)
                dot = xp.dot(self.state.flatten(), v)  # i,ij->j
                probability = xp.real(xp.conj(dot) * dot)
                distribution = xp.random.multinomial(n_trial, probability, size=1)
                ret = xp.sum(w * distribution) / n_trial
                return self._to_scalar(xp.real(ret))

    def _to_scalar(self, x):
        if self.xp != np:
            if isinstance(x, self.xp.ndarray):
                x = self.xp.asnumpy(x)
        if isinstance(x, np.ndarray):
            x = x.item(0)
        return x
