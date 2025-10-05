from parse import parser
import torch
import os
from generators.generator import Seq2SeqGeneratorAllUser
from trainers.sequence_trainer import SeqTrainer
from utils.utils import set_seed
from utils.logger import Logger

args = parser.parse_args()

def stage_1():

    torch.autograd.set_detect_anomaly(True)

    set_seed(args.seed) # fix the random seed
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    args.keepon_path = os.path.join(args.output_dir, args.keepon_path)
    args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    generator = Seq2SeqGeneratorAllUser(args, logger, device)

    trainer = SeqTrainer(args, logger, writer, device, generator)
    trainer.train()

    log_manager.end_log()   # delete the logger threads


def main():

    if args.stage == 1:
        stage_1()
    # elif args.stage == 2:
        # stage_2()
    # else:
        # stage_3()
    return



if __name__ == "__main__":

    main()

