import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

# 数据集定义
class BinarySequenceDataset(Dataset):
    def __init__(self, num_samples, seq_len=10, vec_len=32):
        self.seq_len = seq_len
        self.data = torch.randint(0, 2, (num_samples, seq_len, vec_len), dtype=torch.float32)
        # zero_vector = torch.zeros(num_samples, 1, vec_len)  # 添加一个全零向量作为初始输入
        # self.data = torch.cat((zero_vector, self.data), dim=1)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # 输入和输出是相同的

# 编码器定义
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            # nn.ReLU(),
        )
        self.rnn = nn.GRU(hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        # self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        mlp_output = self.mlp(src)
        # print(mlp_output.size())
        outputs, hidden = self.rnn(self.dropout(mlp_output))
        return hidden

# 解码器定义
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout, device):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hid_dim),
            # nn.ReLU(),
        )
        self.rnn = nn.GRU(hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        # self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Sequential(
            nn.Linear(hid_dim, OUTPUT_DIM*2),
        )

        self.softmax = nn.Softmax(dim=3)
        self.dropout = nn.Dropout(dropout)

        self.rnn_hid_dim = hid_dim
        self.device = device

    def forward(self, trg, hidden, start=False):
        
        batch_size = trg.shape[0]

        if self.training:
            mlp_output = self.mlp(trg)
            start_token = torch.zeros(batch_size, 1, self.rnn_hid_dim, device=self.device)   # 创建一个全为0的向量，形状为 [batch_size, 1, hidden_size], 作为开始向量
            rnn_input = torch.cat((start_token, mlp_output[:, :-1]), dim=1) # 在第2维度（axis=1）的开始处concat全为0的向量
        else:
            if start:
                start_token = torch.zeros(batch_size, 1, self.rnn_hid_dim, device=self.device)
                rnn_input = start_token
            else:
                mlp_output = self.mlp(trg)
                rnn_input = mlp_output

        outputs, hidden = self.rnn(self.dropout(rnn_input), hidden)
        outputs = self.fc_out(outputs)
        outputs = outputs.view(outputs.size(0), outputs.size(1), OUTPUT_DIM, 2)
        predictions = self.softmax(outputs)

        return predictions, hidden

# Seq2Seq模型定义
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    def forward(self, src, trg):
        hidden = self.encoder(src)
        output, _ = self.decoder(trg, hidden)
        return output

    def generate(self, src, max_len):
        hidden = self.encoder(src)

        trg = torch.zeros((src.size(0), 1, src.size(2))).to(self.device) 
        decoder_outputs = torch.zeros(src.size(0), max_len, src.size(2))

        for i in range(0, max_len):
            output, hidden = self.decoder(trg, hidden, i==0)
            output_binary = torch.argmax(output, dim=3).float()
            decoder_outputs[:,i] = output_binary
            trg = output_binary

        return decoder_outputs
    
    def encode(self, src):
        hidden = self.encoder(src)
        
        return hidden
    
    def decode(self, src, max_len, hidden):
        trg = torch.zeros((src.size(0), 1, src.size(2))).to(self.device) 
        decoder_outputs = torch.zeros(src.size(0), max_len, src.size(2)).to(self.device)

        for i in range(0, max_len):
            output, hidden = self.decoder(trg, hidden, i==0)
            output_binary = torch.argmax(output, dim=3).float()
            decoder_outputs[:,i] = output_binary.squeeze(1)
            trg = output_binary

        return decoder_outputs

# 超参数定义
INPUT_DIM = 32
OUTPUT_DIM = 32
HID_DIM = 512
N_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 128
N_EPOCHS = 10
SEQ_LEN = 10

class trainer():

    def __init__(self, load_model=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_save_path = os.path.join(self.base_dir, "seq2seq_model_32bits.pth")
        
        self.train_dataset = BinarySequenceDataset(num_samples=100_000, seq_len=SEQ_LEN, vec_len=INPUT_DIM)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.val_dataset = BinarySequenceDataset(num_samples=1000, seq_len=SEQ_LEN, vec_len=INPUT_DIM)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.test_dataset = BinarySequenceDataset(num_samples=1000, seq_len=SEQ_LEN, vec_len=INPUT_DIM)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        print('Prepare Dataset Finished')

        # 初始化模型、优化器和损失函数
        self.enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, DROPOUT)
        self.dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DROPOUT, device=self.device)
        self.model = Seq2Seq(self.enc, self.dec, self.device).to(self.device)
        
        if load_model and os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path))
        else:
            self.model.apply(self.init_weights)
    
    def init_weights(self, m):
        if hasattr(m, 'weight') and m.weight is not None:
            # 对所有有权重的层使用 Xavier uniform 初始化
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
            # 偏置初始化为 0
            nn.init.zeros_(m.bias)
    
    def retrain(self):

    # 数据加载
        print('Start Training: ')

        optimizer = optim.Adam(self.model.parameters())
        # criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()

        # if os.path.exists(self.model_save_path):
        #     self.model.load_state_dict(torch.load(self.model_save_path))
            
        phases = ['train', 'val']

        # 训练过程
        for epoch in range(N_EPOCHS):
            self.model.train()
            epoch_train_loss = 0
            epoch_train_acc = 0
            epoch_val_loss = 0
            epoch_val_acc = 0
            for phase in phases:
                if phase == 'train':
                    self.model.train()
                    for src, trg in self.train_dataloader:
                        src, trg = src.to(self.device), trg.to(self.device)

                        optimizer.zero_grad()
                        output = self.model(src, trg)
                        # print(trg.size())

                        batch_size, seq_len, output_dim = trg.size()
                        trg_expanded = trg.unsqueeze(-1).repeat(1, 1, 1, 2)
                        trg_expanded[..., 0] = (trg_expanded[..., 0] < 0.5).float()
                        trg_expanded[..., 1] = (trg_expanded[..., 1] >= 0.5).float()

                        loss = criterion(output, trg_expanded)

                        loss.backward()
                        optimizer.step()

                        epoch_train_loss += loss.item()
                        output_binary = (output > 0.5).int()
                        correct_predictions = (output_binary == trg_expanded).float()
                        accuracy = correct_predictions.prod(dim=-1).mean()
                        epoch_train_acc += accuracy

                    print(f'Epoch: {epoch+1:02}, {phase}, Loss: {epoch_train_loss / len(self.train_dataloader):.4f}, Acc: {epoch_train_acc / len(self.train_dataloader):.4f}')
                elif phase == 'val':
                    self.model.eval()
                    for src, trg in self.val_dataloader:
                        src, trg = src.to(self.device), trg.to(self.device)

                        batch_size, seq_len, output_dim = trg.size()
                        trg_expanded = trg.unsqueeze(-1).repeat(1, 1, 1, 2)
                        trg_expanded[..., 0] = (trg_expanded[..., 0] < 0.5).float()
                        trg_expanded[..., 1] = (trg_expanded[..., 1] >= 0.5).float()

                        with torch.no_grad():
                            output = self.model(src,trg)
                            loss = criterion(output, trg_expanded)

                        epoch_val_loss += loss.item()
                        output_binary = (output > 0.5).int()
                        correct_predictions = (output_binary == trg_expanded).float()
                        accuracy = correct_predictions.prod(dim=-1).mean()
                        epoch_val_acc += accuracy

                    print(f'Epoch: {epoch+1:02}, {phase}, Loss: {epoch_val_loss / len(self.val_dataloader):.4f}, Acc: {epoch_val_acc / len(self.val_dataloader):.4f}')

    def test(self, test_exist_model=False):
        print("Start testing: ")
        # 测试模型
        if test_exist_model and os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path))
        
        self.model.eval()
        output_txt_dir = os.path.join(self.base_dir, "output.txt")
        with torch.no_grad():
            def vec2string(vec):
                res = ''
                for i in range(vec.shape[0]):
                    res += str(vec[i]) + ' '
                return res
            with open(output_txt_dir, "w") as f:
                f.write('Test results: \n')
            for i in range(5):  # 打印5个测试样本
                src, trg = self.test_dataset[i]
                src, trg = src.unsqueeze(0).to(self.device), trg.unsqueeze(0).to(self.device)
                output = self.model.generate(src, trg.shape[1])

                # print("Input: ", src.squeeze(0).cpu().numpy())
                # print("Output:", output.squeeze(0).cpu().numpy().round())
                with open(output_txt_dir, "a") as f:
                    Input = src.squeeze(0).cpu().numpy()
                    Output = output.squeeze(0).cpu().numpy().round()
                    f.write(f'Test {i + 1}: \n')
                    for idx in range(Input.shape[0]):
                        f.write(f"Input:  {vec2string(Input[idx])}\n")
                        f.write(f"Output: {vec2string(Output[idx])}\n")
                        f.write(f"acc: {(((Output[idx] > 0.5).astype(int)) == Input[idx]).astype(float).mean()}\n")
                        f.write("-" * 20 + '\n')
                    acc = (((Output > 0.5).astype(int)) == Input).astype(float).mean()
                    f.write(f'Acc: {acc}\n')
                    f.write("="*20 + '\n')
                # print("-" * 20)
                
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

def use_en_decoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DROPOUT, device=device)
    seq2seq_model = Seq2Seq(encoder, decoder, device=device)
    
    model_path = os.path.join(os.path.dirname(__file__), 'seq2seq_model_32bits.pth')
    seq2seq_model.load_state_dict(torch.load(model_path))
    
    for param in seq2seq_model.parameters():
        param.requires_grad = False
    
    batch_size = 30 
    seq_len=10
    vec_len=32
    src = torch.randint(0, 2, (batch_size, seq_len, vec_len), dtype=torch.float32).to(device)
    encoded_hidden = seq2seq_model.encode(src)
    
    max_len = seq_len
    
    generated_output = seq2seq_model.decode(src, max_len, encoded_hidden)
    
    generated_output  = ((generated_output - 0.5) > 0).to(torch.int)
    
    print(f"The accuracy is {torch.mean((generated_output == src).to(torch.float)):.2f}")


# if __name__ == "__main__":
#     my_trainer = trainer(load_model=True)
# #     my_trainer.retrain()
# #     my_trainer.save_model()
#     my_trainer.test()

if __name__ == "__main__":
    use_en_decoder()