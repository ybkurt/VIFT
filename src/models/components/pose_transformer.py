from torch import nn
import torch
import math
class PoseTransformer(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super(PoseTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        seq_length = visual_inertial_features.size(1)

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_features.device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)

        return output


class TokenizedPoseTransformer(nn.Module):
    def __init__(self,
                 input_dim=768,
                 embedding_dim=128,
                 num_layers=2,
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0.1,
                 scale=1.0,
                 low_limit=-1.0,
                 high_limit=1.0,
                 n_tokens=4096,
                 n_special_tokens=1,
                 pad_token_id=0,
                 eos_token_id=1,
                 use_eos_token=False,
                 context_length=11):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, embedding_dim) for k in range(6)])
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.linears = nn.ModuleList([nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, n_tokens)) for k in range(6)])

        from src.models.components.chronos_models.chronos import OdometryBins, ChronosConfig

        tokenizer_config = ChronosConfig(n_tokens=n_tokens,
                                         n_special_tokens=n_special_tokens,
                                         context_length=context_length,
                                         pad_token_id=pad_token_id,
                                         eos_token_id=eos_token_id,
                                         use_eos_token=use_eos_token
                                         )

        self.tokenizer = OdometryBins(low_limit=low_limit,
                                      high_limit=high_limit,
                                      config=tokenizer_config)
        self.tokenizer.centers =  self.tokenizer.centers.to("cuda")
        self.tokenizer.boundaries = self.tokenizer.boundaries.to("cuda")
        self.scale = torch.Tensor([scale]).to("cuda")
        self.CELoss = torch.nn.CrossEntropyLoss()
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def argmax_sampler(self, output_probabilities):
        """
        Sample indices from output probabilities using argmax.
        
        Parameters:
        - output_probabilities (torch.Tensor): A tensor containing output probabilities or logits.
          Shape: [batch_size, sequence_length, num_classes]
        
        Returns:
        - torch.Tensor: A tensor containing the sampled indices (token IDs).
          Shape: [batch_size, sequence_length]
        """
        # Use argmax to get the index of the maximum probability for each token in the sequence
        sampled_indices = torch.argmax(output_probabilities, dim=-1)
    
        return sampled_indices

    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        seq_length = visual_inertial_features.size(1)
        B,S,E = visual_inertial_features.shape
        gt = torch.tensor(gt).view(B,S,6).to("cuda")

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding


        B,S,D = gt.shape

        # Generate input tokens
        _tmp = [self.tokenizer.input_transform(gt[:,:,k], scale=self.scale) for k in range(D)]
        # Separate the outputs into individual components
        tokens, attention_masks, scales = zip(*_tmp)
        # Stack each component separately
        tokens = torch.stack(tokens)  # Shape: (6, batch_size, seq_len)
        tokens = tokens.permute(1, 2, 0)  # Shape: (batch_size, seq_len, 6)
        # tokens (batch_size, seq_len, 6)

        input_tokens = torch.roll(tokens, 1, 1) # shift inputs to right to estimate next tokens from previous ones
        input_tokens[:,0,:] = 0 # make first element of every sequence <PAD> token, due to shifting it contained last pose

        # (batch_size, seq_len,D) -> (batch_size, seq_len, embed_size)
        #TODO: Make embeddings list, multiple embeddings via indexingö work here is done, implement in init
        input_embeddings = torch.stack([self.embeddings[k](input_tokens[:,:,k]) for k in range(D)]) 
        input_embeddings = torch.sum(input_embeddings,dim=0)
        input_embeddings += pos_embedding
        

        # concatenate latents with inputs
        tf_input = torch.cat([visual_inertial_features, input_embeddings], dim=1)
        # tf_input (batch_size, 2*seq_len, embed_size)

        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(int(seq_length*2), visual_inertial_features.device)
        output = self.transformer_encoder(tf_input, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        logits = torch.stack([self.linears[k](output[:,seq_length:,:]) for k in range(D)])  # [6, B, S, vocab_size]
        logits = logits.permute(1, 2, 0, 3)  # Shape: [B, S, 6, vocab_size]

        B, K, T = tokens.shape
        ce = torch.zeros([], device=tokens.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  
            tokens_k = tokens[:, k, ...].contiguous().view(-1)  
            q_ce = self.CELoss(logits_k, tokens_k)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        ce = ce / K

        sampled_indices = self.argmax_sampler(logits) # (batch_size, seq_len,6, num_tokens) -> (batch_size, seq_len,6)

        poses = torch.stack([self.tokenizer.output_transform(sampled_indices[:,:,k], scale=self.scale) for k in range(D)])
        poses = poses.permute(1,2,0) # (6, batch_size, seq_len) -> (batch_size, seq_len, 6)
        assert poses.shape[-1] == 6, 'you need to give 6 dim output'

        return poses, ce.mean()




class PoseTransformerVisual(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        visual_inertial_features = visual_inertial_features[:,:,:512]
        seq_length = visual_inertial_features.size(1)

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_features.device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)

        return output


class PoseTransformerInertial(nn.Module):
    def __init__(self, input_dim=256, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        visual_inertial_features = visual_inertial_features[:,:,512:]
        seq_length = visual_inertial_features.size(1)

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_features.device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)

        return output
