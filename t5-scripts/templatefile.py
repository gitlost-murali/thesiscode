class TemplateHandler0():
  """
  Class to convert equations to semantic parse tree, prefix to suffix etc.
  """
  def __init__(self):
    self.labelmapper = {"OFF": "the comment is offensive", "NOT": "the comment is not offensive"}

  def decode_preds(self, sentence):
    sentence = sentence.lower()
    if not sentence in self.labelmapper.values():
      sentence = "wrong-pred"
    return sentence