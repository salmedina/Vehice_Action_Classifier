scene_video_ids_list = [['VIRAT_S_:000000','VIRAT_S_000001','VIRAT_S_000002','VIRAT_S_000003','VIRAT_S_000004','VIRAT_S_000005','VIRAT_S_000006','VIRAT_S_000007', 'VIRAT_S_000008'],
                    ['VIRAT_S_000101','VIRAT_S_000102'],
                    ['VIRAT_S_000200','VIRAT_S_000201','VIRAT_S_000202','VIRAT_S_000203','VIRAT_S_000204','VIRAT_S_000205','VIRAT_S_000206','VIRAT_S_000207'],
                    ['VIRAT_S_010000','VIRAT_S_010001','VIRAT_S_010002','VIRAT_S_010003','VIRAT_S_010004','VIRAT_S_010005'],
                    ['VIRAT_S_010100','VIRAT_S_010101','VIRAT_S_010102','VIRAT_S_010103','VIRAT_S_010104','VIRAT_S_010105','VIRAT_S_010106','VIRAT_S_010107','VIRAT_S_010108','VIRAT_S_010109','VIRAT_S_010110','VIRAT_S_010111','VIRAT_S_010112','VIRAT_S_010113','VIRAT_S_010114','VIRAT_S_010115','VIRAT_S_010116'],
                    ['VIRAT_S_010200','VIRAT_S_010201','VIRAT_S_010202','VIRAT_S_010203','VIRAT_S_010204','VIRAT_S_010205','VIRAT_S_010206','VIRAT_S_010207','VIRAT_S_010208'],
                    ['VIRAT_S_040000','VIRAT_S_040001','VIRAT_S_040002','VIRAT_S_040003','VIRAT_S_040004','VIRAT_S_040005','VIRAT_S_040104'],
                    ['VIRAT_S_040100','VIRAT_S_040101','VIRAT_S_040102','VIRAT_S_040103'],
                    ['VIRAT_S_050000'],
                    ['VIRAT_S_050100', 'VIRAT_S_050101'],
                    ['VIRAT_S_050200', 'VIRAT_S_050201', 'VIRAT_S_050202', 'VIRAT_S_050203', 'VIRAT_S_050204'],
                    ['VIRAT_S_050300', 'VIRAT_S_050301'],
                    ['G32700', 'G32701', 'G32702', 'G32703', 'G32704', 'G32705', 'G32706','G32707', 'G32708'],
                    ['G32709', 'G32710', 'G32711', 'G32712', 'G32713', 'G32714', 'G32715', 'G32716', 'G32717','G32718', 'G32719', 'G32720'],
                    ['G32800', 'G32801', 'G32802', 'G32803', 'G32804', 'G32805', 'G32806', 'G32807', 'G32808', 'G32809', 'G32810', 'G32811', 'G32812', 'G32813'],
                    ['G33000', 'G33001'],
                    ['G33600'],
                    ['G33700', 'G33701', 'G33702', 'G33703', 'G33704', 'G33705', 'G33706', 'G33707', 'G33708', 'G33709', 'G33710', 'G33711', 'G33712', 'G33713', 'G33714'],
                    ['G34100', 'G34101', 'G34102', 'G34103', 'G34104', 'G34105', 'G34106', 'G34107', 'G34108', 'G34109', 'G34110', 'G34111', 'G34112']]

def get_num_scenes():
    return len(scene_video_ids_list)

def get_video_ids(sid):
    '''
    Gets the list of video ids based on the scene id [0,num_scenes)
    :param sid: scene id
    :return: list of video ids
    '''
    if sid not in range(get_num_scenes()):
        return []

    return scene_video_ids_list[sid]