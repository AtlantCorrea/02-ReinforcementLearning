
class PID_ctrl():
    def __init__(self, kp, ki, kd) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sum = 0
        self.dt = 20/1000*5
        self.error_a = None
        pass

    def step(self, error):
        prop = self.kp * error

        self.sum += self.ki*(error*self.dt)
        inte = self.sum 

        if self.error_a is None:
            self.error_a = error
        deri = self.kd*(error - self.error_a)/self.dt
        self.error_a = error

        return prop + inte + deri
