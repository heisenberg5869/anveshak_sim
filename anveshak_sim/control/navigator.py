class Navigator:
    def decide(self, obstacles):
        if not obstacles:
            return "FORWARD"

        left = sum(1 for _, y in obstacles if y > 0)
        right = sum(1 for _, y in obstacles if y < 0)

        # If blocked straight ahead → STOP
        if left == right and left > 0:
            return "STOP"

        if left > right:
            return "RIGHT"
        else:
            return "LEFT"
