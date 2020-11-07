

def get_username_and_password(path="./username.txt"):
    """ Get username/password from file. Just returns a list, one item per line in credential file
    Args:
        path:

    Returns:
    """

    with open(str(path), "r") as pw:
        return [x.strip() for x in pw.read().split()]


