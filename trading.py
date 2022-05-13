class Position:

    def __init__(self, str_type, price, volume):
        if str_type == 'long':
            self.type = 'long'
        elif str_type == 'short':
            self.type = 'short'
        else:
            raise ValueError('Trying to open unknown position type')
        self.p_open = price
        self.vol = volume
        self.lifetime = 0

    def hold(self):
        self.lifetime += 1

    def close(self, price):
        if self.type == 'long':
            return self.vol * (price - self.p_open)
        if self.type == 'short':
            return self.vol * (self.p_open - price)
        raise ValueError('Tyring to close unknown position type')



