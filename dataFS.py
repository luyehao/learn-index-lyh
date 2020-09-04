import os
import logging
from config import CONFIG


class DataFS():
    ITEMLEN = CONFIG['ITEMLEN']
    KEYLEN = CONFIG["KEYLEN"]
    FILENAME=CONFIG["FILENAME"]
    BLOCKSIZE=CONFIG["BLOCKSIZE"]

    def __init__(self):
        if not os.path.exists(self.FILENAME):
            with open(self.FILENAME, 'w', encoding='utf-8'):
                pass
        self.file = open(self.FILENAME, 'r+t', encoding='utf-8')
        self.file.seek(0)
        buf = self.file.read(self.ITEMLEN)
        self.first_item=buf.find('\n')
        self.file.seek(0)
        self.first_key, self.first_value = self._splitKV(self.file.read(self.first_item))
        filesize=os.path.getsize(self.FILENAME)
        self.file.seek(filesize-self.ITEMLEN)
        buf=self.file.read(self.ITEMLEN)
        self.last_item=filesize-self.ITEMLEN+buf[:-1].rfind('\n')+1
        self.file.seek(self.last_item)
        self.last_key,self.last_value=self._splitKV(self.file.read(self.ITEMLEN))

    def BS(self, arr, x) -> (str, str):
        l = 0
        r = len(arr) - 1
        while (r > l):
            mid = int(l + (r - l) / 2)
            k,v = self._splitKV(arr[mid])
            if k == x:
                return mid
            elif k > x:
                r = mid
            else:
                l = mid + 1
        return r

    def _splitKV(self,buf:str)->(str,str):
        if buf.count(' ') != 1:
            logging.warning("not valid KV pair:{}".format(buf))
            return buf, ''
        return buf.split(' ')[0],buf.split(' ')[1]

    def seek(self,blocks,key,offset)->(str,str):
        if blocks.find('\n')==self.first_item:
            begin=0
        else:
            begin = blocks.find('\n') + 1
        end = blocks.rfind('\n')
        kvs = blocks[begin:end].split('\n')
        idx = self.BS(kvs, key)
        k, v = self._splitKV(kvs[idx])
        if idx==0 and k>key:
            if k==self.first_key:
                return k,v
            self.file.seek(max(offset - 2*self.BLOCKSIZE, 0))
            pre_buf=self.file.read(2*self.BLOCKSIZE)
            return self.seek(pre_buf,key,offset-self.BLOCKSIZE)
        elif idx==len(kvs)-1 and k<key:
            if k==self.last_key:
                return k,v
            self.file.seek(max(offset, 0))
            nxt_buf = self.file.read(2 * self.BLOCKSIZE)
            return self.seek(nxt_buf, key, offset+self.BLOCKSIZE)
        return k,v

    def query(self,key,offset)->(str,str):
        self.file.seek(max(offset - self.BLOCKSIZE, 0))
        buf = self.file.read(2 * self.BLOCKSIZE)
        if len(buf)<self.ITEMLEN:
            return self.query(key,self.last_item)
        return self.seek(buf,key,offset)

    def gen_train_data(self, isheader=True) -> (list, list):
        """
        :param isheader:是否返回记录头
        :return:X是字符串  Y是整数
        """
        offset = 0
        self.file.seek(0)
        X = []
        Y = []
        with open('raw_index', 'w') as f:
            for line in self.file.readlines():
                k = line[:line.find(' ')]
                X.append(k)
                Y.append(offset+len(line) / 2 if isheader else len(line))
                f.write("{}\t{}\n".format(k, offset))
                offset += len(line)
        return X, Y

if __name__ == '__main__':
    dfs=DataFS()
    print(dfs.query('ffd47e5fa6add6ea1909ca',40000))