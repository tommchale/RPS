from pickle import FRAME
import random
import cv2
from keras.models import load_model
import numpy as np
import time
import statistics


class RPS:
    def __init__(self):
        # loads the tenable model
        self.model = load_model('keras_model.h5')
        # opens the camera and defines a capture object
        self.cap = cv2.VideoCapture(0)
        self.rps_options = ['rock', 'scissors', 'paper', 'nothing']
        self.computer_win_count = 0
        self.user_win_count = 0
        # defines shape of data array
        # an Array of 1 dimensions 224*224*3
        # numpy.ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)[source]
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Captures the video frame by frame
        # Define text details
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 0, 255)
        self.thickness = 2
        self.message_center = ""
        self.message_upper_center = ""
        self.message_lower_center = ""
        self.message_lower_right = ""
        self.message_lower_left = ""
        self.exit_flag = False

    def run_camera(self):

        ret, self.frame = self.cap.read()

        cv2.putText(self.frame, self.message_center, (100, 200), self.font,
                    self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        cv2.putText(self.frame, self.message_upper_center, (100, 100), self.font,
                    self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        cv2.putText(self.frame, self.message_lower_center, (100, 300), self.font,
                    self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        cv2.putText(self.frame, self.message_lower_left, (100, 400), self.font,
                    self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        cv2.putText(self.frame, self.message_lower_right, (100, 500), self.font,
                    self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        # Resize image
        #self.frame = cv2.resize(self.frame, (480, 270))
        resized_frame = cv2.resize(
            self.frame, (224, 224), interpolation=cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) /
                            127.0) - 1  # Normalize the image
        self.data[0] = normalized_image
        # shows the image from the frame read by the camera
        cv2.imshow('Camera Open', self.frame)

    def get_computer_choice(self):

        self.computer_choice = random.choice(self.rps_options[:3])

    def get_user_choice(self):

        while True:
            self.intro()
            self.present_choice()
            self.compute_prediction()
            break

    def intro(self):

        self.message_upper_center = ""
        self.message_lower_center = ""
        self.message_center = "Press 'q' to quit or 'c' to play RPS!"

        while True:
            self.run_camera()
            # Press c to continue the window
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                self.exit_flag = True
                break

    def present_choice(self):
        self.message_upper_center = "Please present choice in..."
        timeout = time.time() + 4
        while (timeout - time.time()) > 0:
            self.run_camera()
            self.message_center = (
                f"{timeout - time.time():.0f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def compute_prediction(self):

        predicition_list = []
        timeout = time.time() + 0.5
        while (timeout - time.time()) > 0:

            self.run_camera()
            index = self.get_prediction()
            predicition_list.append(index)

        mode_index = statistics.mode(predicition_list)
        self.user_choice = self.rps_options[mode_index]

    def get_prediction(self):
        prediction = self.model.predict(self.data)
        index = np.argmax(prediction[0])
        return index

    def compute_round_winner(self):

        if self.user_choice == 'nothing':
            self.winner = 'none'

        if self.computer_choice == self.user_choice:
            self.winner = 'nothing'
            self.message_lower_right = (
                f"That was a draw. The score is Computer: {self.computer_win_count} - User: {self.user_win_count}")

        elif (self.computer_choice == 'paper' and self.user_choice == 'rock') or (self.computer_choice == 'scissors' and self.user_choice == 'paper') or (self.computer_choice == 'rock' and self.user_choice == 'scissors'):
            self.winner = 'computer'

        elif self.user_choice != 'nothing':
            self.winner = 'user'

    def compute_total_score(self):

        self.message_lower_left = (
            f" User: {self.user_choice} Computer: {self.computer_choice}")

        if self.winner == 'none':
            self.message_lower_right = (
                f"Please show an option and try again")

        if self.winner == 'computer':
            self.computer_win_count += 1
            self.message_lower_right = (
                f"The score is Computer: {self.computer_win_count} - User: {self.user_win_count}")

        if self.winner == 'user':
            self.user_win_count += 1
            self.message_lower_right = (
                f"The score is Computer: {self.computer_win_count} - User: {self.user_win_count}")

    def check_for_winner(self):

        if self.computer_win_count == 2:
            self.message_lower_center = (
                f"Unlucky the computer won {self.computer_win_count} : {self.user_win_count} ")
            self.game_over()

        elif self.user_win_count == 2:
            self.message_lower_center = (
                f"Congratulations you beat the computer {self.user_win_count} : {self.computer_win_count} ")
            self.game_over()

    def game_over(self):

        self.message_center = "Press 'p' to play again, or 'q' to quit."
        self.message_lower_right = ""
        self.message_upper_center = ""

        while True:

            self.run_camera()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.exit_flag = True
                break
            elif cv2.waitKey(1) & 0xFF == ord('p'):
                self.computer_win_count = 0
                self.user_win_count = 0
                self.message_lower_left = ""
                break

    def close_and_destroy_windows(self):
        # After the loop release the cap object
        (self.cap).release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    def play_game(self):

        while self.exit_flag == False:

            self.get_computer_choice()
            self.get_user_choice()
            self.compute_round_winner()
            self.compute_total_score()
            self.check_for_winner()

        self.close_and_destroy_windows()


def play():
    game = RPS()
    game.play_game()


play()
