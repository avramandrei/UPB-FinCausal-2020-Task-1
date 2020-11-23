import torch


def load_data(path, tokenizer, device, batch_size=32, shuffle=False):
    list_tokens = []
    list_masks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            sentence = line.replace("\n", "")
            tokens = tokenizer.encode(sentence,
                                      add_special_tokens=True,
                                      do_lower_case=False)

            list_tokens.append(torch.tensor(tokens))
            list_masks.append(torch.tensor([1] * (len(tokens))))

    # pad the tokens, the labels and the masks
    X = torch.nn.utils.rnn.pad_sequence(list_tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

    masks = torch.nn.utils.rnn.pad_sequence(list_masks, batch_first=True, padding_value=0).to(device)

    # create the loader
    dataset = torch.utils.data.TensorDataset(X, masks)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader