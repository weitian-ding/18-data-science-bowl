from keras import backend as K

SMOOTH = 1.


def dice_coef_loss(y_true, y_pred):
    return -_dice_coef(y_true, y_pred)


def _dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)


def custom_metrics_dict():
    return {
        'dice_coef_loss': dice_coef_loss
    }