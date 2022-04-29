import random
from secrets import choice


class RPS:

    def __init__(self):
        self.rps_options = ['rock', 'paper', 'scissors']

    def get_computer_choice(self):

        self.computer_choice = random.choice(self.rps_options)

        return self.computer_choice

    def get_user_choice(self):

        self.user_choice = input(
            ' \n Please chose rock, paper or scissors : \n')
        self.user_choice = self.user_choice.lower()

        return self.user_choice

    def get_winner(self):
        if self.computer_choice == self.user_choice:
            print('draw')

        elif (self.computer_choice == 'paper' and self.user_choice == 'rock') or (self.computer_choice == 'scissors' and self.user_choice == 'paper') or (self.computer_choice == 'rock' and self.user_choice == 'scissors'):
            self.winner = 'computer'
            print(self.winner)
            return self.winner

        else:
            self.winner = 'user'
            print(self.winner)
            return self.winner


def play():
    game = RPS()
    game.get_computer_choice()
    game.get_user_choice()
    game.get_winner()


play()
