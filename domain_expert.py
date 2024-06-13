import torch
import transformers


def forward(self, hidden_states: torch.Tensor, force_experts: torch.Tensor) -> torch.Tensor:
    assert self.top_k > 1, "top_k should be greater than 1 for domain expert routing"
    batch_size, sequence_length, hidden_dim = hidden_states.shape

    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

    # Select top_k-1 experts based on routing_weights
    _, selected_experts_top_k_minus_1 = torch.topk(routing_weights, self.top_k - 1, dim=-1)

    # Expand force_experts to match the shape of selected_experts_top_k_minus_1
    force_experts_expanded = force_experts.view(-1, 1).expand(-1, self.top_k - 1)

    # Concatenate force_experts with selected_experts_top_k_minus_1
    selected_experts = torch.cat((force_experts_expanded, selected_experts_top_k_minus_1), dim=-1)

    # Gather the corresponding routing_weights for the selected experts
    routing_weights = routing_weights.gather(1, selected_experts)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Rest of the code remains the same
    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

def monkey_patch_domain_expert_routing_forward_function():
    transformers.models.mixtral.modeling_mixtral.forward = forward


if __name__ == "__main__":
    monkey_patch_domain_expert_routing_forward_function()
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Conversation
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    chatbot = pipeline(model=model, tokenizer=tokenizer)
    conversation = Conversation("I'm looking for a movie - what's your favourite one?")
    conversation = chatbot(conversation)
    print(conversation.messages[-1]["content"])
    conversation.add_message({"role": "user", "content": "That's interesting, why do you like action movies?"})
    conversation = chatbot(conversation)
    print(conversation.messages[-1]["content"])

