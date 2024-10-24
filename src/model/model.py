import torch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoConfig, AutoModel

__all__ = ["DetectModel", "ClassifyModel"]


class DetectModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.use_hidden_states = config.use_hidden_states
        self.model_config = AutoConfig.from_pretrained(config.model_path)
        self.model_config.update(
            {
                "hidden_dropout_prob": config.hidden_dropout,
                "attention_probs_dropout_prob": config.attention_dropout,
                "output_hidden_states": True,
            }
        )
        hidden_size = self.model_config.hidden_size
        self.backbone = AutoModel.from_pretrained(config.model_path, config=self.model_config)

        self.lstm_type = config.lstm_type
        if config.lstm_type == "lstm":
            self.lstm = nn.LSTM(
                hidden_size * self.use_hidden_states, hidden_size, num_layers=1, batch_first=True, bidirectional=True
            )
        elif config.lstm_type == "gru":
            self.lstm = nn.GRU(
                hidden_size * self.use_hidden_states, hidden_size, num_layers=1, batch_first=True, bidirectional=True
            )

        self.pos_emb = nn.Sequential(
            nn.Linear(2, hidden_size * self.use_hidden_states),
            nn.Dropout(config.dropout),
        )
        head_input_size = hidden_size * self.use_hidden_states if config.lstm_type == "none" else hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(head_input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.class_num),
        )
        self.layer_norm = nn.LayerNorm(hidden_size * self.use_hidden_states)
        self.dropout = nn.Dropout(config.dropout)

        self.head.apply(self._init_weights)
        if config.lstm_type != "none":
            self._lstm_init_weights(self.lstm)

    # DeBERTa weight initialization function
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Tensorflow/Keras-like initialization for GRU
    def _lstm_init_weights(self, module):
        for name, p in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name:
                p.data.fill_(0)

    def forward(self, input_ids, attention_mask, positions):
        x_pos = self.pos_emb(positions)
        x_bb = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x_bb = torch.cat(x_bb.hidden_states[-self.use_hidden_states :], dim=-1)
        x = x_bb + x_pos
        x = self.layer_norm(x)
        if self.lstm_type != "none":
            x, _ = self.lstm(x)
            x = self.dropout(x)
        x = self.head(x)
        return x

    # Reinitialize the specified number of layers (counting from the last encoder layer)
    def reinit_layers(self, reinit_layer_num: int):
        for i in range(1, reinit_layer_num + 1):
            self.backbone.encoder.layer[-i].apply(self._init_weights)

    # Freeze the specified number of layers (counting from the first embedding layer)
    def freeze_layers(self, freeze_layer_num: int):
        for i in range(freeze_layer_num):
            if i == 0:
                for params in self.backbone.embeddings.parameters():
                    params.requires_grad = False
            else:
                for params in self.backbone.encoder.layer[i - 1].parameters():
                    params.requires_grad = False

    # Freeze all layers except the specified number of reinitialized layers
    def freeze_backbone(self, reinit_layer_num: int):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for i in range(1, reinit_layer_num + 1):
            for params in self.backbone.encoder.layer[-i].parameters():
                params.requires_grad = True

    # Unfreeze the backbone, keeping the originally frozen layers frozen
    def unfreeze_backbone(self, freeze_layer_num: int):
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.freeze_layers(freeze_layer_num)


class ClassifyModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.use_hidden_states = config.use_hidden_states
        self.model_config = AutoConfig.from_pretrained(config.model_path)
        self.model_config.update(
            {
                "hidden_dropout_prob": config.hidden_dropout,
                "attention_probs_dropout_prob": config.attention_dropout,
                "output_hidden_states": True,
            }
        )
        hidden_size = self.model_config.hidden_size
        self.backbone = AutoModel.from_pretrained(config.model_path, config=self.model_config)

        self.lstm_type = config.lstm_type
        if config.lstm_type == "lstm":
            self.lstm = nn.LSTM(
                hidden_size * self.use_hidden_states, hidden_size, num_layers=1, batch_first=True, bidirectional=True
            )
        elif config.lstm_type == "gru":
            self.lstm = nn.GRU(
                hidden_size * self.use_hidden_states, hidden_size, num_layers=1, batch_first=True, bidirectional=True
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * self.use_hidden_states, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )
        self.layer_norm = nn.LayerNorm(hidden_size * self.use_hidden_states)
        self.dropout = nn.Dropout(config.dropout)

        self.head.apply(self._init_weights)

    # DeBERTa weight initialization function
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.cat(outputs.hidden_states[-self.use_hidden_states :], dim=-1)  # N層分のhidden_statesをconcat
        if self.lstm_type != "none":
            x, _ = self.lstm(x)
            x = self.dropout(x)
        x = self.head(x)
        return x.squeeze(-1)

    # Reinitialize the specified number of layers (counting from the last encoder layer)
    def reinit_layers(self, reinit_layer_num: int):
        for i in range(1, reinit_layer_num + 1):
            self.backbone.encoder.layer[-i].apply(self._init_weights)

    # Freeze the specified number of layers (counting from the first embedding layer)
    def freeze_layers(self, freeze_layer_num: int):
        for i in range(freeze_layer_num):
            if i == 0:
                for params in self.backbone.embeddings.parameters():
                    params.requires_grad = False
            else:
                for params in self.backbone.encoder.layer[i - 1].parameters():
                    params.requires_grad = False

    # Freeze all layers except the specified number of reinitialized layers
    def freeze_backbone(self, reinit_layer_num: int):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for i in range(1, reinit_layer_num + 1):
            for params in self.backbone.encoder.layer[-i].parameters():
                params.requires_grad = True

    # Unfreeze the backbone, keeping the originally frozen layers frozen
    def unfreeze_backbone(self, freeze_layer_num: int):
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.freeze_layers(freeze_layer_num)
