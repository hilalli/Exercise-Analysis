class LSTM_ModelSettings:
    def __init__(self,
                 dataset_path,
                 dataset_sequence_length,
                 dataset_resize_width,
                 dataset_resize_height,
                 dataset_color_channels,
                 dataset_validation_ratio,
                 dataset_test_ratio,
                 
                 model_filter_counts,
                 model_kernel_sizes,
                 model_activations,
                 model_pool_sizes,
                 model_lstm_unit_counts,
                 model_dense_unit_counts,
                 model_dropout_unit_counts,
                 
                 compiling_scheduler_epoch_patience,
                 compiling_scheduler_learning_rate_multiplier,
                 compiling_early_stopping_monitor,
                 compiling_early_stopping_patience,
                 compiling_early_stopping_mode_restore_best_weights,
                 compiling_learning_rate,
                 compiling_loss,
                 compiling_metrics,
                 
                 training_epoch_count,
                 training_batch_size,
                 training_shuffle,
                 
                 statistics_graph_size):
        
        self.DATASET_PATH = dataset_path
        self.DATASET_SEQUENCE_LENGTH = dataset_sequence_length
        self.DATASET_RESIZE_WIDTH = dataset_resize_width
        self.DATASET_RESIZE_HEIGHT = dataset_resize_height
        self.DATASET_COLOR_CHANNELS = dataset_color_channels
        self.DATASET_VALIDATION_RATIO = dataset_validation_ratio
        self.DATASET_TEST_RATIO = dataset_test_ratio
            
        self.MODEL_FILTER_COUNTS = model_filter_counts
        self.MODEL_KERNEL_SIZES = model_kernel_sizes
        self.MODEL_ACTIVATIONS = model_activations
        self.MODEL_POOL_SIZES = model_pool_sizes
        self.MODEL_LSTM_UNIT_COUNTS = model_lstm_unit_counts
        self.MODEL_DENSE_UNIT_COUNTS = model_dense_unit_counts
        self.MODEL_DROPOUT_UNIT_COUNTS = model_dropout_unit_counts
           
        self.COMPILING_SCHEDULER_EPOCH_PATIENCE = compiling_scheduler_epoch_patience
        self.COMPILING_SCHEDULER_LEARNING_RATE_MULTIPLIER = compiling_scheduler_learning_rate_multiplier
        self.COMPILING_EARLY_STOPPING_MONITOR = compiling_early_stopping_monitor
        self.COMPILING_EARLY_STOPPING_PATIENCE = compiling_early_stopping_patience
        self.COMPILING_EARLY_STOPPING_MODE_RESTORE_BEST_WEIGHTS = compiling_early_stopping_mode_restore_best_weights
        self.COMPILING_LEARNING_RATE = compiling_learning_rate
        self.COMPILING_LOSS = compiling_loss
        self.COMPILING_METRICS = compiling_metrics
        
        self.TRAINING_EPOCH_COUNT = training_epoch_count
        self.TRAINING_BATCH_SIZE = training_batch_size
        self.TRAINING_SHUFFLE = training_shuffle
        
        self.STATISTICS_GRAPH_SIZE = statistics_graph_size
        
        return
    

    def getting_dataset_path(self):
        return self.DATASET_PATH
    
    def getting_dataset_sequence_length(self):
        return self.DATASET_SEQUENCE_LENGTH
    
    def getting_dataset_resize_width(self):
        return self.DATASET_RESIZE_WIDTH
    
    def getting_dataset_resize_height(self):
        return self.DATASET_RESIZE_HEIGHT
    
    def getting_dataset_color_channel_count(self):
        return self.DATASET_COLOR_CHANNELS
    
    def getting_dataset_validation_ratio(self):
        return self.DATASET_VALIDATION_RATIO
    
    def getting_dataset_test_ratio(self):
        return self.DATASET_TEST_RATIO
    

    def getting_model_filter_count_for_layer(self, layer_index):
        return self.MODEL_FILTER_COUNTS[layer_index]
    
    def getting_model_kernel_size_for_layer(self, layer_index):
        return self.MODEL_KERNEL_SIZES[layer_index]

    def getting_model_activation_for_layer(self, layer_index):
        return self.MODEL_ACTIVATIONS[layer_index]
    
    def getting_model_pool_size_for_layer(self, layer_index):
        return self.MODEL_POOL_SIZES[layer_index]
    
    def getting_model_lstm_unit_count_for_layer(self, layer_index):
        return self.MODEL_LSTM_UNIT_COUNTS[layer_index]
    
    def getting_model_dense_unit_count_for_layer(self, layer_index):
        return self.MODEL_DENSE_UNIT_COUNTS[layer_index]
    
    def getting_model_dropout_unit_count_for_layer(self, layer_index):
        return self.MODEL_DROPOUT_UNIT_COUNTS[layer_index]
    

    def getting_compiling_scheduler_epoch_patience(self):
        return self.COMPILING_SCHEDULER_EPOCH_PATIENCE
    
    def getting_compiling_scheduler_learning_rate_multiplier(self):
        return self.COMPILING_SCHEDULER_LEARNING_RATE_MULTIPLIER
    
    def getting_compiling_checkpoint_monitor(self):
        return self.COMPILING_CHECKPOINT_MONITOR

    def getting_compiling_checkpoint_save_weights_only(self):
        return self.COMPILING_CHECKPOINT_SAVE_WEIGHTS_ONLY

    def getting_compiling_early_stopping_monitor(self):
        return self.COMPILING_EARLY_STOPPING_MONITOR

    def getting_compiling_early_stopping_patience(self):
        return self.COMPILING_EARLY_STOPPING_PATIENCE
    
    def getting_compiling_early_stopping_mode_restore_best_weights(self):
        return self.COMPILING_EARLY_STOPPING_MODE_RESTORE_BEST_WEIGHTS
    
    def getting_compiling_learning_rate(self):
        return self.COMPILING_LEARNING_RATE
    
    def getting_compiling_loss(self):
        return self.COMPILING_LOSS
    
    def getting_compiling_metrics(self):
        return self.COMPILING_METRICS
    

    def getting_training_epoch_count(self):
        return self.TRAINING_EPOCH_COUNT
    
    def getting_training_batch_size(self):
        return self.TRAINING_BATCH_SIZE
    
    def getting_training_shuffle(self):
        return self.TRAINING_SHUFFLE
    

    def getting_statistics_graph_size(self):
        return self.STATISTICS_GRAPH_SIZE
    
