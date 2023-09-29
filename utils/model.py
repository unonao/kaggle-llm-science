import peft
import torch
from typing import Optional, Tuple, Union

from torch import nn
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, PeftModel
from torch import FloatTensor, LongTensor
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import (
    MultipleChoiceModelOutput,
)


class LlamaMultipleChoice(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = LlamaModel(config)
        hidden_size: int = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1, bias=False)
        self.init_weights()

    def set_lora(self, peft_config):
        self.backbone: PeftModel = peft.get_peft_model(self.model, peft_config)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.backbone.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.backbone(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_length: LongTensor = flat_attention_mask.sum(dim=1)
        eos_hidden_states: FloatTensor = outputs.last_hidden_state[
            torch.arange(
                seq_length.size(0),
                device=outputs.last_hidden_state.device,
            ),
            seq_length - 1,
        ]
        logits = self.classifier(eos_hidden_states)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
