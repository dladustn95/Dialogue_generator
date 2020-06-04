import os
from torchtext import data, datasets

PAD, BOS, EOS = 1, 2, 3

class DataLoader():
    def __init__(self,
                 train_src_fn=None,
                 train_tgt_fn=None,
                 valid_src_fn=None,
                 valid_tgt_fn=None,
                 batch_size=64,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=True,
                 dsl=False
                 ):

        super(DataLoader, self).__init__()

        self.src = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>' if dsl else None,
                              eos_token='<EOS>' if dsl else None
                              )

        self.tgt = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>' if use_bos else None,
                              eos_token='<EOS>' if use_eos else None
                              )

        if train_src_fn is not None and train_tgt_fn is not None:
            train = TranslationDataset(src_path=train_src_fn,
                                       tgt_path=train_tgt_fn,
                                       fields=[('src', self.src),
                                               ('tgt', self.tgt)
                                               ],
                                       max_length=max_length
                                       )
        if valid_src_fn is not None and valid_tgt_fn is not None:
            valid = TranslationDataset(src_path=valid_src_fn,
                                       tgt_path=valid_tgt_fn,
                                       fields=[('src', self.src),
                                               ('tgt', self.tgt)
                                               ],
                                       max_length=max_length
                                       )

            self.train_iter = data.BucketIterator(train,
                                                  batch_size=batch_size,
                                                  device='cuda:%d' % device if device >= 0 else 'cpu',
                                                  shuffle=shuffle,
                                                  sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=True
                                                  )
            self.valid_iter = data.BucketIterator(valid,
                                                  batch_size=batch_size,
                                                  device='cuda:%d' % device if device >= 0 else 'cpu',
                                                  shuffle=False,
                                                  sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=True
                                                  )

            self.src.build_vocab(train, max_size=max_vocab)
            self.tgt.build_vocab(train, max_size=max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, src_path, tgt_path, fields, max_length=None, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(tgt_path, encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()),
                                                   len(trg_line.split())
                                                   ):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super().__init__(examples, fields, **kwargs)


if __name__ == '__main__':
    import sys
    loader = DataLoader(sys.argv[1],
                        sys.argv[2],
                        (sys.argv[3], sys.argv[4]),
                        batch_size=8
                        )

    print(len(loader.src.vocab))
    print(len(loader.tgt.vocab))

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.src)
        print(batch.tgt)

        if batch_index > 1:
            break