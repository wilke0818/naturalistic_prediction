from simple_model import SimpleSurfacePredictor
from plis_model import PlisSimpleSurfacePredictor
from dice_model import DiceSurfacePredictor
from sparse_dice_model import DiceSparseSurfacePredictor
from mini_dice_model import MiniDiceSurfacePredictor

#model_class = SimpleSurfacePredictor
model_class = PlisSimpleSurfacePredictor
#model_class = DiceSurfacePredictor
#model_class = DiceSparseSurfacePredictor
#model_class = MiniDiceSurfacePredictor

model_testing_classes = {
        MiniDiceSurfacePredictor.MODEL_NAME: MiniDiceSurfacePredictor,
        DiceSparseSurfacePredictor.MODEL_NAME: DiceSparseSurfacePredictor,
        DiceSurfacePredictor.MODEL_NAME: DiceSurfacePredictor,
        PlisSimpleSurfacePredictor.MODEL_NAME: PlisSimpleSurfacePredictor,
        SimpleSurfacePredictor.MODEL_NAME: SimpleSurfacePredictor
}
