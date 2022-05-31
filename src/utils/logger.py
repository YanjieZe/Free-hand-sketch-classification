from termcolor import colored

class Logger:
    """
    Logger class
    """

    def __init__(self, log_file_path):
        """
        Constructor
        """
        self.log_file_path = log_file_path

    def log(self, message):
        """
        Logs a message
        """
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(message + '\n')
        # color print
        print(colored(message, 'red'))
    