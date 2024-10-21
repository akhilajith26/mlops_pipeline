import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load your model
model = tf.keras.models.load_model("model.h5")

# Apply pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruned_model = prune_low_magnitude(model)

# Save the pruned model
pruned_model.save("pruned_model.h5")
