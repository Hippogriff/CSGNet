"""Utility function for tweaking learning rate on the fly"""


class LearningRate:
    """
    utils functions to manipulate the learning rate
    """

    def __init__(self,
                 optimizer,
                 init_lr=0.001,
                 lr_dacay_fact=0.2,
                 patience=10,
                 logger=None):
        """
        :param logger: Logger to output stuff into file.
        :param optimizer: Object of the torch optimizer initialized before
        :param init_lr: Start lr
        :param lr_decay_epoch: Epchs after which the learning rate to be decayed
        :param lr_dacay_fact: Factor by which lr to be decayed
        :param patience: Number of epochs to wait for the loss to decrease
        before reducing the lr
        """
        self.opt = optimizer
        self.init_lr = init_lr
        self.lr_dacay_fact = lr_dacay_fact
        self.loss = 1e8
        self.patience = patience
        self.pat_count = 0
        self.lr = init_lr
        self.logger = logger
        pass

    def red_lr_by_fact(self):
        """
        reduces the learning rate by the pre-specified factor
        :return:
        """
        # decay factor lesser than one.
        self.lr = self.lr * self.lr_dacay_fact
        for param_group in self.opt.param_groups:
            param_group['lr'] = self.lr
        if self.logger:
            self.logger.info('LR is set to {}'.format(self.lr))
        else:
            print('LR is set to {}'.format(self.lr))

    def reduce_on_plateu(self, loss):
        """
        Reduce the learning rate when loss doesn't decrease
        :param loss: loss to be monitored
        :return: optimizer with new lr
        """
        if self.loss > loss:
            self.loss = loss
            self.pat_count = 0
        else:
            self.pat_count += 1
            if self.pat_count > self.patience:
                self.pat_count = 0
                self.red_lr_by_fact()
