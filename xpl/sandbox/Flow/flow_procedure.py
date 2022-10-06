


class SupervisedFlowTrainingProcedure(TrainingProcedure):

    def __init__(self,
                 model_management_unit,
                 models,
                 task_name):
        super(SupervisedFlowTrainingProcedure, self).__init__(
            model_management_unit=model_management_unit,
            models=models,
            task_name=task_name
        )
        self.flow_loss = SupervisedOpticalFlowLoss().to(self.device)

    def init_data_loaders(self):
        train_dataset = SintelDataset(home_dir='/data/sintel/',
                                      image_size=128,
                                      random_transform=True)

        eval_dataset = SintelDataset(home_dir='/data/sintel/',
                                     image_size=128,
                                     random_transform=False)

        data_loaders = XPLDataLoader(datasets={'train': train_dataset,
                                               'eval': eval_dataset},
                                     batch_size=32
                                     )

        return data_loaders

    def forward_batch(self, batch):
        first_frames = batch['first_frames'].to(self.device)
        second_frames = batch['second_frames'].to(self.device)
        target_flows = batch['flows'].to(self.device)

        (predicted_flows_1_to_2, _,
         occlusion_1_to_2, _) = self.models['flow_network'](first_frames=first_frames,
                                                            second_frames=second_frames)
        (_, predicted_flows_2_to_1,
         _, occlusion_2_to_1) = self.models['flow_network'](first_frames=second_frames,
                                                            second_frames=first_frames)

        measurements_1_to_2 = self.flow_loss(first_frames=first_frames,
                                             second_frames=second_frames,
                                             predicted_flows=predicted_flows_1_to_2,
                                             target_flows=target_flows,
                                             predicted_occlusions=occlusion_1_to_2)

        measurements_2_to_1 = self.flow_loss(first_frames=second_frames,
                                             second_frames=first_frames,
                                             predicted_flows=predicted_flows_2_to_1,
                                             target_flows=target_flows,
                                             predicted_occlusions=occlusion_2_to_1)
        return {
            'loss': {
                k: measurements_1_to_2['loss'][k]+measurements_2_to_1['loss'][k]
                for k in measurements_1_to_2['loss'].keys()
            }
        }

    def visualize(self):
        batch = self.data_loaders.get_next_batch(input_set='train')

        first_frames = batch['first_frames'].to(self.device)
        second_frames = batch['second_frames'].to(self.device)

        (predicted_flows_1_to_2,
         predicted_flows_2_to_1,
         predicted_occlusions_1_to_2,
         predicted_occlusions_2_to_1) = self.models['flow_network'](first_frames=first_frames,
                                                                    second_frames=second_frames)

        rgb_1_to_2 = self.flow_to_rgb(predicted_flows_1_to_2)
        rgb_2_to_1 = self.flow_to_rgb(predicted_flows_2_to_1)

        projected_first_frames = self.flow_loss.project(frames=second_frames, flows=predicted_flows_2_to_1)
        projected_second_frames = self.flow_loss.project(frames=first_frames, flows=predicted_flows_1_to_2)

        import matplotlib.pyplot as plt
        for i in range(first_frames.shape[0]):
            plt.subplot(2, 4, 1)
            plt.imshow(first_frames[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 2)
            plt.imshow(projected_first_frames[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 3)
            plt.imshow(rgb_2_to_1[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 4)
            plt.imshow(torch.exp(predicted_occlusions_1_to_2[i]).permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 5)
            plt.imshow(second_frames[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 6)
            plt.imshow(projected_second_frames[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 7)
            plt.imshow(rgb_1_to_2[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 8)
            plt.imshow(torch.exp(predicted_occlusions_2_to_1[i]).permute(1, 2, 0).detach().cpu().numpy())

            plt.show()

    def flow_to_rgb(self, flow):
        r = torch.sqrt(flow[:, 0:1, :, :]** 2 + flow[:, 1:2, :, :]** 2 + 1e-5)
        return torch.cat([
            10*r,
            (10*flow + 0.5),
        ], dim=1)
        



class UnsupervisedFlowTrainingProcedure(TrainingProcedure):

    def __init__(self,
                 model_management_unit,
                 models,
                 task_name):
        super(UnsupervisedFlowTrainingProcedure, self).__init__(
            model_management_unit=model_management_unit,
            models=models,
            task_name=task_name
        )
        self.flow_loss = UnsupervisedOpticalFlowLoss().to(self.device)

    def init_data_loaders(self):
        train_dataset = YFCCFlowDataset(home_dir='/data/yfcc100m/data/videos/keyframes/',
                                        image_size=128,
                                        random_transform=True)

        eval_dataset = YFCCFlowDataset(home_dir='/data/yfcc100m/data/videos/keyframes/6c8',
                                       image_size=128,
                                       random_transform=True)

        data_loaders = XPLDataLoader(datasets={'train': train_dataset,
                                               'eval': eval_dataset},
                                     batch_size=32
                                     )
        return data_loaders

    def forward_batch(self, batch):

        first_frames = batch['first_frames'].to(self.device)
        second_frames = batch['second_frames'].to(self.device)

        (predicted_flows_1_to_2,
         predicted_flows_2_to_1,
         predicted_occlusions_1_to_2,
         predicted_occlusions_2_to_1) = self.models['flow_network'](first_frames=first_frames,
                                                                    second_frames=second_frames)

        return self.flow_loss(first_frames=first_frames,
                              second_frames=second_frames,
                              flows_1_to_2=predicted_flows_1_to_2,
                              flows_2_to_1=predicted_flows_2_to_1,
                              occlusions_1_to_2=predicted_occlusions_1_to_2,
                              occlusions_2_to_1=predicted_occlusions_2_to_1)

    def visualize(self):
        batch = self.data_loaders.get_next_batch(input_set='train')

        first_frames = batch['first_frames'].to(self.device)
        second_frames = batch['second_frames'].to(self.device)

        (predicted_flows_1_to_2,
         predicted_flows_2_to_1,
         predicted_occlusions_1_to_2,
         predicted_occlusions_2_to_1) = self.models['flow_network'](first_frames=first_frames,
                                                                    second_frames=second_frames)

        rgb_1_to_2 = self.flow_to_rgb(predicted_flows_1_to_2)
        rgb_2_to_1 = self.flow_to_rgb(predicted_flows_2_to_1)

        projected_first_frames = self.flow_loss.project(frames=second_frames, flows=predicted_flows_2_to_1)
        projected_second_frames = self.flow_loss.project(frames=first_frames, flows=predicted_flows_1_to_2)

        import matplotlib.pyplot as plt
        for i in range(first_frames.shape[0]):
            plt.subplot(2, 4, 1)
            plt.imshow(first_frames[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 2)
            plt.imshow(projected_first_frames[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 3)
            plt.imshow(rgb_2_to_1[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 4)
            plt.imshow(torch.exp(predicted_occlusions_1_to_2[i]).permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 5)
            plt.imshow(second_frames[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 6)
            plt.imshow(projected_second_frames[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 7)
            plt.imshow(rgb_1_to_2[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(2, 4, 8)
            plt.imshow(torch.exp(predicted_occlusions_2_to_1[i]).permute(1, 2, 0).detach().cpu().numpy())

            plt.show()

    def flow_to_rgb(self, flow):
        r = torch.sqrt(flow[:, 0:1, :, :]** 2 + flow[:, 1:2, :, :]** 2 + 1e-5)
        return torch.cat([
            r,
            (flow + 0.5),
        ], dim=1)
        
