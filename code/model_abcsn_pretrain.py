import keras
from keras.layers import Reshape, Dense
from keras_hub.layers import SinePositionEncoding, TransformerEncoder


def make_model(
    num_wvl,
    intermediate_dim=256,
    num_heads=8,
    do_enc=0.50,
):
    inputs = keras.Input(shape=num_wvl)
    x = inputs

    x = Reshape((1, num_wvl))

    x = Dense(1024, activation="linear")(x)
    x = Reshape((64, 16))(x)
    x = Dense(128, activation="relu")(x)

    PE = SinePositionEncoding()(x)
    PE = Dense(64, activation="gelu")(PE)
    PE = Dense(x.shape[2], activation="linear")(PE)
    x = x + PE

    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True,
    )(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True,
    )(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True,
    )(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True,
    )(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True,
    )(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True,
    )(x)

    x = Reshape((num_wvl,))(x)
    outputs = Dense(num_wvl, activation="linear", use_bias=False)(x)
    model = keras.Model(inputs, outputs)

    return model
