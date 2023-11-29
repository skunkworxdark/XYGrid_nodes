from invokeai.app.invocations.baseinvocation import (
  BaseInvocation,
  InputField,
  invocation,
)
from invokeai.app.invocations.primitives import FloatOutput

def process(text: str) -> float:
  text = text.strip()
  if text.endswith('%'):
    formattedText = text[:-1].strip()
    number = float(formattedText) / 100
  else:
    number = float(text)
  return number

@invocation(
  'percent_to_float',
  title='Percent to Float',
  tags=[
    'convert',
    'float',
    'normalize',
    'number',
    'percent',
    'percentage',
    'rescale',
    'scale',
    'text',
  ],
  category='string',
  version='1.0.0',
)
class PercentToFloatInvocation(BaseInvocation):
  """Converts a string to a float. If the input ends with a percent sign, it will be rescaled accordingly."""
  text: str = InputField(
    title='Text',
    description='Input text',
  )

  def invoke(self, context) -> FloatOutput:
    output = process(self.text)
    return FloatOutput(value=output)
