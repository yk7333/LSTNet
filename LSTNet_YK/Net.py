import torch
from torch import nn


class LSTNet_AttSkip(nn.Module):
    def __init__(self, params):
        super(LSTNet_AttSkip, self).__init__()
        self.data_dim = params.data.shape[1]  # 数据的特征数
        self.window_size = params.window_size  # 窗口的大小
        self.kernel_size = params.kernel_size  # 卷积核大小 (具体来说是核大小的第0维，第1维度是数据的特征数data_dim)
        self.hiddenC = params.hiddenC  # CNN隐藏神经元数目
        self.hiddenG = params.hiddenG  # GRU隐藏神经元数目
        self.hiddenS = params.hiddenS  # GRUSkip隐藏神经元数目
        self.T = params.T  # 跳跃周期T
        self.ar_window = params.ar_window  # AutoRegressive的窗口大小
        self.dropout = nn.Dropout(p=params.dropout)  # dropout大小
        self.device = "cuda:0"
        self.s_window = self.window_size // self.T + 1  # 跳跃层的窗口大小 将以为T周期的数据维度作为一个window,窗口大小为=data_dim//T+1
        self.attention = nn.Linear(self.hiddenG, 1)     #注意力层
        self.softmax = nn.Softmax(dim=1)
        self.LSTM = nn.LSTMCell(self.hiddenC, self.hiddenG)
        self.cnn = nn.Conv2d(1, self.hiddenC, kernel_size=(self.kernel_size, self.data_dim))  # CNN层
        self.gru = nn.GRU(self.hiddenC, self.hiddenG)  # GRU层
        self.gru_skip = nn.GRU(self.hiddenC, self.hiddenS)  # GRUskip层
        self.linear_g = nn.Linear(self.hiddenG, self.data_dim)
        self.linear_s = nn.Linear(self.hiddenS * self.T, self.data_dim)
        self.linear_a = nn.Linear(self.ar_window, 1)
    def forward(self, x):
        # CNN层
        batch_size = x.shape[0]  # x:(batch_size,window_size,data_dim)
        c = torch.unsqueeze(x, 1)  # c:(batch_size,1,window_size,data_dim)
        c = c.permute(0, 1, 3, 2)  # c:(batch_size,1,data_dim,window_size)
        c = c.reshape(batch_size * self.data_dim, self.window_size)  # c:(batch_size*data_dim,window_size)
        pad = torch.zeros(c.shape[0], self.kernel_size - 1, device=self.device)  # pad:(batch_size,kernel_size-1)
        c = torch.cat((pad, c), 1)  # 左向padding操作 c:(batch_size*data_dim,window_size+kernel_size-1)
        c = c.reshape(batch_size, 1, self.data_dim,
                      self.window_size + self.kernel_size - 1)  # c:（batch_size,1,data_dim,window_size+kernel_size-1)
        c = c.permute(0, 1, 3, 2)  # c:（batch_size,1,window_size+kernel_size-1,data_dim)
        c = self.cnn(c)  # c:(batch_size,hiddenC,window_size,1)
        c = torch.relu(c)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)  # c:(batch_size,hiddenC,window_size)

        # 注意力机制的GRU层
        g = c.permute(2, 0, 1)  # g:(window_size,batch_size,hiddenC)
        out = torch.zeros(self.window_size - 1, batch_size, self.hiddenG)  # out:(window_size-1,batch_size,hiddenG)
        C = torch.zeros(batch_size, self.hiddenG)  # C:(batch_size,hiddenG)
        h = torch.zeros(batch_size, self.hiddenG)  # h:(batch_size,hiddenG)
        out, h, C = out.cuda(), h.cuda(), C.cuda()
        for i in range(self.window_size - 1):
            h, C = self.LSTM(g[i], (h, C))
            out[i] = h  # out保存t-p到t-1时刻的隐状态h
        out = out.permute(1, 0, 2)  # out:(batch_size,window_size-1,hiddenG)
        # 注意力机制
        att_out = self.attention(out)  # att_out:(batch_size,window_size-1,1)
        att_out = self.softmax(att_out)  # att_out:(batch_size,window_size-1,1)
        att_out = (out * att_out)  # att_out:(batch_size,window_size-1,hiddenG)
        att_out = att_out.sum(1)  # att_out:(batch_size,hiddenG)
        att_out = att_out + out[:, -1, :]  # att_out:(batch_size,hiddenG)
        h, _ = self.LSTM(att_out, (h, C))  # h:(batch_size,hiddenG)
        res = self.linear_g(h)  # res:(batch_size,data_dim)
        res = self.dropout(res)

        # GRUskip层
        s = c.reshape(batch_size * self.hiddenC, self.window_size)  # s:(batch_size*hiddenC,window_size)
        # 补0让新的window_size变为T*s_window,即一共有s_window个包含T个数据的数据段
        pad = torch.zeros(batch_size * self.hiddenC, self.T * self.s_window - self.window_size, device=self.device)
        s = torch.cat((pad, s), 1)  # s:(batch_size*hiddenC,T*s_window)
        s = s.reshape(batch_size, self.hiddenC, self.T, self.s_window)  # s:(batch_size,hiddenC,T,s_window)
        s = s.permute(3, 0, 2, 1)  # s:(s_window,batch_size,T,hiddenC)
        s = s.reshape(self.s_window, batch_size * self.T, self.hiddenC)  # s:(s_window,batch_size*T,hiddenC)
        _, h2 = self.gru_skip(s)  # h2:(1,batch_size*T,hiddenS)
        h2 = self.dropout(h2)
        h2 = h2.reshape(batch_size, self.T * self.hiddenS)  # h2:(batch_size,T*hiddenS)  论文中的Σh_S(t-i)
        res = res + self.linear_s(h2)  # res:(batch_size,data_dim)

        # AR层
        # AR层取window中后ar_window个数据作为输入数据
        a = x[:, -self.ar_window:, :]  # a:(batch_size,ar_window,data_dim)
        a = a.permute(0, 2, 1)  # a:(batch_size,data_dim,ar_window)
        a = a.reshape(batch_size * self.data_dim, self.ar_window)  # a:(batch_size*data_dim,ar_window)
        a = self.linear_a(a)  # a:(batch_size*data_dim,1)
        a = self.dropout(a)
        a = a.reshape(batch_size, self.data_dim)  # a:(batch_size,data_dim)
        res = res + a  # res:(batch_size,data_dim)

        return res


class LSTNet_Att(nn.Module):
    def __init__(self, params):
        super(LSTNet_Att, self).__init__()
        self.data_dim = params.data.shape[1]  # 数据的特征数
        self.window_size = params.window_size  # 窗口的大小
        self.kernel_size = params.kernel_size  # 卷积核大小 (具体来说是核大小的第0维，第1维度是数据的特征数data_dim)
        self.hiddenC = params.hiddenC  # CNN隐藏神经元数目
        self.hiddenG = params.hiddenG  # GRU隐藏神经元数目
        self.ar_window = params.ar_window  # AutoRegressive的窗口大小
        self.dropout = nn.Dropout(p=params.dropout)  # dropout大小
        self.device = "cuda:0"
        self.cnn = nn.Conv2d(1, self.hiddenC, kernel_size=(self.kernel_size, self.data_dim))  # CNN层
        self.linear_g = nn.Linear(self.hiddenG, self.data_dim)
        self.linear_a = nn.Linear(self.ar_window, 1)
        self.attention = nn.Linear(self.hiddenG, 1)
        self.softmax = nn.Softmax(dim=1)
        self.LSTM = nn.LSTMCell(self.hiddenC, self.hiddenG)

    def forward(self, x):
        # CNN层
        batch_size = x.shape[0]  # x:(batch_size,window_size,data_dim)
        c = torch.unsqueeze(x, 1)  # c:(batch_size,1,window_size,data_dim)
        c = c.permute(0, 1, 3, 2)  # c:(batch_size,1,data_dim,window_size)
        c = c.reshape(batch_size * self.data_dim, self.window_size)  # c:(batch_size*data_dim,window_size)
        pad = torch.zeros(c.shape[0], self.kernel_size - 1, device=self.device)  # pad:(batch_size,kernel_size-1)
        c = torch.cat((pad, c), 1)  # 左向padding操作 c:(batch_size*data_dim,window_size+kernel_size-1)
        c = c.reshape(batch_size, 1, self.data_dim,
                      self.window_size + self.kernel_size - 1)  # c:（batch_size,1,data_dim,window_size+kernel_size-1)
        c = c.permute(0, 1, 3, 2)  # c:（batch_size,1,window_size+kernel_size-1,data_dim)
        c = self.cnn(c)  # c:(batch_size,hiddenC,window_size,1)
        c = torch.relu(c)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)  # c:(batch_size,hiddenC,window_size)

        # 注意力机制的GRU层
        g = c.permute(2, 0, 1)  # g:(window_size,batch_size,hiddenC)
        out = torch.zeros(self.window_size - 1, batch_size, self.hiddenG)  # out:(window_size-1,batch_size,hiddenG)
        C = torch.zeros(batch_size, self.hiddenG)  # C:(batch_size,hiddenG)
        h = torch.zeros(batch_size, self.hiddenG)  # h:(batch_size,hiddenG)
        out, h, C = out.cuda(), h.cuda(), C.cuda()
        for i in range(self.window_size - 1):
            h, C = self.LSTM(g[i], (h, C))
            out[i] = h  # out保存t-p到t-1时刻的隐状态h
        out = out.permute(1, 0, 2)  # out:(batch_size,window_size-1,hiddenG)
        # 注意力机制
        att_out = self.attention(out)  # att_out:(batch_size,window_size-1,1)
        att_out = self.softmax(att_out)  # att_out:(batch_size,window_size-1,1)
        att_out = (out * att_out)  # att_out:(batch_size,window_size-1,hiddenG)
        att_out = att_out.sum(1)  # att_out:(batch_size,hiddenG)
        att_out = att_out + out[:, -1, :]  # att_out:(batch_size,hiddenG)
        h, _ = self.LSTM(att_out, (h, C))  # h:(batch_size,hiddenG)
        res = self.linear_g(h)  # res:(batch_size,data_dim)
        res = self.dropout(res)

        # AR层
        # AR层取window中后ar_window个数据作为输入数据
        a = x[:, -self.ar_window:, :]  # a:(batch_size,ar_window,data_dim)
        a = a.permute(0, 2, 1)  # a:(batch_size,data_dim,ar_window)
        a = a.reshape(batch_size * self.data_dim, self.ar_window)  # a:(batch_size*data_dim,ar_window)
        a = self.linear_a(a)  # a:(batch_size*data_dim,1)
        a = self.dropout(a)
        a = a.reshape(batch_size, self.data_dim)  # a:(batch_size,data_dim)
        res = res + a  # res:(batch_size,data_dim)

        return res


class LSTNet_Skip(nn.Module):
    def __init__(self, params):
        super(LSTNet_Skip, self).__init__()
        self.data_dim = params.data.shape[1]  # 数据的特征数
        self.window_size = params.window_size  # 窗口的大小
        self.kernel_size = params.kernel_size  # 卷积核大小 (具体来说是核大小的第0维，第1维度是数据的特征数data_dim)
        self.hiddenC = params.hiddenC  # CNN隐藏神经元数目
        self.hiddenG = params.hiddenG  # GRU隐藏神经元数目
        self.hiddenS = params.hiddenS  # GRUSkip隐藏神经元数目
        self.T = params.T  # 跳跃周期T
        self.ar_window = params.ar_window  # AutoRegressive的窗口大小
        self.dropout = nn.Dropout(p=params.dropout)  # dropout大小
        self.device = "cuda:0"
        self.s_window = self.window_size // self.T + 1  # 跳跃层的窗口大小 将以为T周期的数据维度作为一个window,窗口大小为=data_dim//T+1
        self.cnn = nn.Conv2d(1, self.hiddenC, kernel_size=(self.kernel_size, self.data_dim))  # CNN层
        self.gru = nn.GRU(self.hiddenC, self.hiddenG)  # GRU层
        self.gru_skip = nn.GRU(self.hiddenC, self.hiddenS)  # GRUskip层
        self.linear_g = nn.Linear(self.hiddenG, self.data_dim)
        self.linear_s = nn.Linear(self.hiddenS * self.T, self.data_dim)
        self.linear_a = nn.Linear(self.ar_window, 1)

    def forward(self, x):
        # CNN层
        batch_size = x.shape[0]  # x:(batch_size,window_size,data_dim)
        c = torch.unsqueeze(x, 1)  # c:(batch_size,1,window_size,data_dim)
        c = c.permute(0, 1, 3, 2)  # c:(batch_size,1,data_dim,window_size)
        c = c.reshape(batch_size * self.data_dim, self.window_size)  # c:(batch_size*data_dim,window_size)
        pad = torch.zeros(c.shape[0], self.kernel_size - 1, device=self.device)  # pad:(batch_size,kernel_size-1)
        c = torch.cat((pad, c), 1)  # 左向padding操作 c:(batch_size*data_dim,window_size+kernel_size-1)
        c = c.reshape(batch_size, 1, self.data_dim,
                      self.window_size + self.kernel_size - 1)  # c:（batch_size,1,data_dim,window_size+kernel_size-1)
        c = c.permute(0, 1, 3, 2)  # c:（batch_size,1,window_size+kernel_size-1,data_dim)
        c = self.cnn(c)  # c:(batch_size,hiddenC,window_size,1)
        c = torch.relu(c)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)  # c:(batch_size,hiddenC,window_size)

        # GRU层
        g = c.permute(2, 0, 1)  # g:(window_size,batch_size,hiddenC)
        _, h1 = self.gru(g)  # h1:(1,batch_size,hiddenG)
        h1 = self.dropout(h1)
        h1 = torch.squeeze(h1, 0)  # h1:(batch_size,hiddenG)  论文中的h_R(t)
        res = self.linear_g(h1)  # res:(batch_size,data_dim)

        # GRUskip层
        s = c.reshape(batch_size * self.hiddenC, self.window_size)  # s:(batch_size*hiddenC,window_size)
        # 补0让新的window_size变为T*s_window,即一共有s_window个包含T个数据的数据段
        pad = torch.zeros(batch_size * self.hiddenC, self.T * self.s_window - self.window_size, device=self.device)
        s = torch.cat((pad, s), 1)  # s:(batch_size*hiddenC,T*s_window)
        s = s.reshape(batch_size, self.hiddenC, self.T, self.s_window)  # s:(batch_size,hiddenC,T,s_window)
        s = s.permute(3, 0, 2, 1)  # s:(s_window,batch_size,T,hiddenC)
        s = s.reshape(self.s_window, batch_size * self.T, self.hiddenC)  # s:(s_window,batch_size*T,hiddenC)
        _, h2 = self.gru_skip(s)  # h2:(1,batch_size*T,hiddenS)
        h2 = self.dropout(h2)
        h2 = h2.reshape(batch_size, self.T * self.hiddenS)  # h2:(batch_size,T*hiddenS)  论文中的Σh_S(t-i)
        res = res + self.linear_s(h2)  # res:(batch_size,data_dim)

        # AR层
        # AR层取window中后ar_window个数据作为输入数据
        a = x[:, -self.ar_window:, :]  # a:(batch_size,ar_window,data_dim)
        a = a.permute(0, 2, 1)  # a:(batch_size,data_dim,ar_window)
        a = a.reshape(batch_size * self.data_dim, self.ar_window)  # a:(batch_size*data_dim,ar_window)
        a = self.linear_a(a)  # a:(batch_size*data_dim,1)
        a = self.dropout(a)
        a = a.reshape(batch_size, self.data_dim)  # a:(batch_size,data_dim)
        res = res + a  # res:(batch_size,data_dim)

        return res


class LSTNet_without_CNN(nn.Module):
    def __init__(self, params):
        super(LSTNet_without_CNN, self).__init__()
        self.data_dim = params.data.shape[1]  # 数据的特征数
        self.window_size = params.window_size  # 窗口的大小
        self.hiddenG = params.hiddenG  # GRU隐藏神经元数目
        self.hiddenS = params.hiddenS  # GRUSkip隐藏神经元数目
        self.T = params.T  # 跳跃周期T
        self.ar_window = params.ar_window  # AutoRegressive的窗口大小
        self.dropout = nn.Dropout(p=params.dropout)  # dropout大小
        self.device = "cuda:0"
        self.s_window = self.window_size // self.T + 1  # 跳跃层的窗口大小 将以为T周期的数据维度作为一个window,窗口大小为=data_dim//T+1
        self.gru = nn.GRU(self.data_dim, self.hiddenG)  # GRU层
        self.gru_skip = nn.GRU(self.data_dim, self.hiddenS)  # GRUskip层
        self.linear_g = nn.Linear(self.hiddenG, self.data_dim)
        self.linear_s = nn.Linear(self.hiddenS * self.T, self.data_dim)
        self.linear_a = nn.Linear(self.ar_window, 1)

    def forward(self, x):
        batch_size = x.shape[0]  # x:(batch_size,window_size,data_dim)
        # GRU层
        g = x.permute(1, 0, 2)  # g:(window_size,batch_size,data_dim)
        _, h1 = self.gru(g)  # h1:(1,batch_size,hiddenG)
        h1 = self.dropout(h1)
        h1 = torch.squeeze(h1, 0)  # h1:(batch_size,hiddenG)  论文中的h_R(t)
        res = self.linear_g(h1)  # res:(batch_size,data_dim)

        # GRUskip层
        s = x.reshape(batch_size * self.data_dim, self.window_size)  # s:(batch_size*data_dim,window_size)
        # 补0让新的window_size变为T*s_window,即一共有s_window个包含T个数据的数据段
        pad = torch.zeros(batch_size * self.data_dim, self.T * self.s_window - self.window_size, device=self.device)
        s = torch.cat((pad, s), 1)  # s:(batch_size*hiddenC,T*s_window)
        s = s.reshape(batch_size, self.data_dim, self.T, self.s_window)  # s:(batch_size,data_dim,T,s_window)
        s = s.permute(3, 0, 2, 1)  # s:(s_window,batch_size,T,hiddenC)
        s = s.reshape(self.s_window, batch_size * self.T, self.data_dim)  # s:(s_window,batch_size*T,data_dim)
        _, h2 = self.gru_skip(s)  # h2:(1,batch_size*T,hiddenS)
        h2 = self.dropout(h2)
        h2 = h2.reshape(batch_size, self.T * self.hiddenS)  # h2:(batch_size,T*hiddenS)  论文中的Σh_S(t-i)
        res = res + self.linear_s(h2)  # res:(batch_size,data_dim)

        # AR层
        # AR层取window中后ar_window个数据作为输入数据
        a = x[:, -self.ar_window:, :]  # a:(batch_size,ar_window,data_dim)
        a = a.permute(0, 2, 1)  # a:(batch_size,data_dim,ar_window)
        a = a.reshape(batch_size * self.data_dim, self.ar_window)  # a:(batch_size*data_dim,ar_window)
        a = self.linear_a(a)  # a:(batch_size*data_dim,1)
        a = self.dropout(a)
        a = a.reshape(batch_size, self.data_dim)  # a:(batch_size,data_dim)
        res = res + a  # res:(batch_size,data_dim)

        return res


class LSTNet_without_Skip(nn.Module):
    def __init__(self, params):
        super(LSTNet_without_Skip, self).__init__()
        self.data_dim = params.data.shape[1]  # 数据的特征数
        self.window_size = params.window_size  # 窗口的大小
        self.kernel_size = params.kernel_size  # 卷积核大小 (具体来说是核大小的第0维，实际上进行的是一维卷积，第1维度是数据的特征数)
        self.hiddenC = params.hiddenC  # CNN隐藏神经元数目
        self.hiddenG = params.hiddenG  # GRU隐藏神经元数目
        self.T = params.T  # 跳跃周期T
        self.ar_window = params.ar_window  # AutoRegressive的窗口大小
        self.dropout = nn.Dropout(p=params.dropout)  # dropout大小
        self.device = "cuda:0"
        self.cnn = nn.Conv2d(1, self.hiddenC, kernel_size=(self.kernel_size, self.data_dim))  # CNN层
        self.gru = nn.GRU(self.hiddenC, self.hiddenG)  # GRU层
        self.linear_g = nn.Linear(self.hiddenG, self.data_dim)
        self.linear_a = nn.Linear(self.ar_window, 1)

    def forward(self, x):
        # CNN层
        batch_size = x.shape[0]  # x:(batch_size,window_size,data_dim)
        c = torch.unsqueeze(x, 1)  # c:(batch_size,1,window_size,data_dim)
        c = c.permute(0, 1, 3, 2)  # c:(batch_size,1,data_dim,window_size)
        c = c.reshape(batch_size * self.data_dim, self.window_size)  # c:(batch_size*data_dim,window_size)
        pad = torch.zeros(c.shape[0], self.kernel_size - 1, device=self.device)  # pad:(batch_size,kernel_size-1)
        c = torch.cat((pad, c), 1)  # 左向padding操作 c:(batch_size*data_dim,window_size+kernel_size-1)
        c = c.reshape(batch_size, 1, self.data_dim,
                      self.window_size + self.kernel_size - 1)  # c:（batch_size,1,data_dim,window_size+kernel_size-1)
        c = c.permute(0, 1, 3, 2)  # c:（batch_size,1,window_size+kernel_size-1,data_dim)
        c = self.cnn(c)  # c:(batch_size,hiddenC,window_size,1)
        c = torch.relu(c)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)  # c:(batch_size,hiddenC,window_size)

        # GRU层
        g = c.permute(2, 0, 1)  # g:(window_size,batch_size,hiddenC)
        _, h1 = self.gru(g)  # h1:(1,batch_size,hiddenG)
        h1 = self.dropout(h1)
        h1 = torch.squeeze(h1, 0)  # h1:(batch_size,hiddenG)  论文中的h_R(t)
        res = self.linear_g(h1)  # res:(batch_size,data_dim)

        # AR层
        # AR层取window中后ar_window个数据作为输入数据
        a = x[:, -self.ar_window:, :]  # a:(batch_size,ar_window,data_dim)
        a = a.permute(0, 2, 1)  # a:(batch_size,data_dim,ar_window)
        a = a.reshape(batch_size * self.data_dim, self.ar_window)  # a:(batch_size*data_dim,ar_window)
        a = self.linear_a(a)  # a:(batch_size*data_dim,1)
        a = self.dropout(a)
        a = a.reshape(batch_size, self.data_dim)  # a:(batch_size,data_dim)
        res = res + a  # res:(batch_size,data_dim)

        return res


class LSTNet_without_AR(nn.Module):
    def __init__(self, params):
        super(LSTNet_without_AR, self).__init__()
        self.data_dim = params.data.shape[1]  # 数据的特征数
        self.window_size = params.window_size  # 窗口的大小
        self.kernel_size = params.kernel_size  # 卷积核大小 (具体来说是核大小的第0维，实际上进行的是一维卷积，第1维度是数据的特征数)
        self.hiddenC = params.hiddenC  # CNN隐藏神经元数目
        self.hiddenG = params.hiddenG  # GRU隐藏神经元数目
        self.hiddenS = params.hiddenS  # GRUSkip隐藏神经元数目
        self.T = params.T  # 跳跃周期T
        self.ar_window = params.ar_window  # AutoRegressive的窗口大小
        self.dropout = nn.Dropout(p=params.dropout)  # dropout大小
        self.device = "cuda:0"
        self.s_window = self.window_size // self.T + 1  # 跳跃层的窗口大小 将以为T周期的数据维度作为一个window,窗口大小为=data_dim//T+1
        self.cnn = nn.Conv2d(1, self.hiddenC, kernel_size=(self.kernel_size, self.data_dim))  # CNN层
        self.gru = nn.GRU(self.hiddenC, self.hiddenG)  # GRU层
        self.gru_skip = nn.GRU(self.hiddenC, self.hiddenS)  # GRUskip层
        self.linear_g = nn.Linear(self.hiddenG, self.data_dim)
        self.linear_s = nn.Linear(self.hiddenS * self.T, self.data_dim)

    def forward(self, x):
        # CNN层
        batch_size = x.shape[0]  # x:(batch_size,window_size,data_dim)
        c = torch.unsqueeze(x, 1)  # c:(batch_size,1,window_size,data_dim)
        c = c.permute(0, 1, 3, 2)  # c:(batch_size,1,data_dim,window_size)
        c = c.reshape(batch_size * self.data_dim, self.window_size)  # c:(batch_size*data_dim,window_size)
        pad = torch.zeros(c.shape[0], self.kernel_size - 1, device=self.device)  # pad:(batch_size,kernel_size-1)
        c = torch.cat((pad, c), 1)  # 左向padding操作 c:(batch_size*data_dim,window_size+kernel_size-1)
        c = c.reshape(batch_size, 1, self.data_dim,
                      self.window_size + self.kernel_size - 1)  # c:（batch_size,1,data_dim,window_size+kernel_size-1)
        c = c.permute(0, 1, 3, 2)  # c:（batch_size,1,window_size+kernel_size-1,data_dim)
        c = self.cnn(c)  # c:(batch_size,hiddenC,window_size,1)
        c = torch.relu(c)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)  # c:(batch_size,hiddenC,window_size)

        # GRU层
        g = c.permute(2, 0, 1)  # g:(window_size,batch_size,hiddenC)
        _, h1 = self.gru(g)  # h1:(1,batch_size,hiddenG)
        h1 = self.dropout(h1)
        h1 = torch.squeeze(h1, 0)  # h1:(batch_size,hiddenG)  论文中的h_R(t)
        res = self.linear_g(h1)  # res:(batch_size,data_dim)

        # GRUskip层
        s = c.reshape(batch_size * self.hiddenC, self.window_size)  # s:(batch_size*hiddenC,window_size)
        # 补0让新的window_size变为T*s_window,即一共有s_window个包含T个数据的数据段
        pad = torch.zeros(batch_size * self.hiddenC, self.T * self.s_window - self.window_size, device=self.device)
        s = torch.cat((pad, s), 1)  # s:(batch_size*hiddenC,T*s_window)
        s = s.reshape(batch_size, self.hiddenC, self.T, self.s_window)  # s:(batch_size,hiddenC,T,s_window)
        s = s.permute(3, 0, 2, 1)  # s:(s_window,batch_size,T,hiddenC)
        s = s.reshape(self.s_window, batch_size * self.T, self.hiddenC)  # s:(s_window,batch_size*T,hiddenC)
        _, h2 = self.gru_skip(s)  # h2:(1,batch_size*T,hiddenS)
        h2 = self.dropout(h2)
        h2 = h2.reshape(batch_size, self.T * self.hiddenS)  # h2:(batch_size,T*hiddenS)  论文中的Σh_S(t-i)
        res = res + self.linear_s(h2)  # res:(batch_size,data_dim)

        return res
