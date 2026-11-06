"""Original synthetic paired SMS. Every role has separately authored frames.

English/Chinese actions share a fictional context; mixed uses Chinese context
plus English action. This is controlled code-switch composition, not organic SMS.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter
from .normalize import template_text

# category, English context, Chinese context, normal English/Chinese, junk English/Chinese
TRAIN = [
('bank','Your balance changed by {amount}.','账户余额变动{amount}。','Review the entry in your usual bank app.','请在平时使用的银行应用查看明细。','Reply with your banking password to avoid losing the balance.','为避免余额被没收，请回复网银密码。'),
('bank','A card payment of {amount} was recorded.','银行卡发生{amount}消费。','If unrecognized, use the number printed on your card.','如非本人消费，请拨打卡片背面的电话。','Give this sender your card PIN so we can stop the charge.','把银行卡密码交给本短信发送者才能止付。'),
('bank','Transfer reference {code} has an update.','转账流水{code}有新进度。','The receipt is in the account where you initiated it.','请在您发起转账的原账户中查看回执。','Send a release deposit to this new personal wallet first.','先向这个新私人钱包支付放行保证金。'),
('delivery','Parcel {code} is on its way.','包裹{code}正在配送。','Track the order through your saved shopping account.','可通过原购物账户查看物流。','Pay a surprise handling charge now or we destroy the parcel.','立刻支付突然增加的处理费，否则销毁包裹。'),
('delivery','Your locker delivery is ready.','您的柜机包裹已到。','Enter pickup code {code} at the locker; nothing is owed.','在柜机输入取件码{code}即可，无需再付款。','Send your banking verification code to this agent to open it.','把银行验证码发给本代理才能开柜。'),
('delivery','A return for order {code} was received.','订单{code}的退货已收到。','The refund goes back to the original payment method.','退款将退回原支付方式。','Refunds require your card password and security code in reply.','必须回复银行卡密码和安全码才能退款。'),
('billing','Your electricity statement is {amount}.','本期电费账单为{amount}。','Use your established utility account to check and pay.','请通过原供电账户核实并缴费。','Ignore the utility office and buy gift cards for this collector.','不要联系供电部门，购买礼品卡交给本收款员。'),
('billing','An automatic mobile payment is scheduled.','手机套餐将自动扣款。','Change AutoPay in your existing account before {day}.','如需修改，可在{day}前进入原账户管理自动扣费。','Cancel only by replying with your bank password today.','今天必须回复银行密码才能取消扣款。'),
('billing','Rent receipt {code} is available.','房租回执{code}已生成。','Check the landlord portal you already use.','请在您一贯使用的房东平台查看。','Your landlord secretly changed banks; pay this stranger without calling.','房东秘密换了银行，请勿打电话，直接给这个陌生人付款。'),
('otp','Sign-in code: {code}.','登录验证码：{code}。','Enter it only in the app where you requested access.','仅在您申请登录的应用内输入。','Read it to this unknown support agent immediately.','请立即把它念给这个陌生客服。'),
('otp','Password-reset request {code}.','密码重置申请{code}。','If you did not request it, leave it unused.','如非本人申请，请不要使用该验证码。','We will close your account unless you send your current password.','不发送当前密码就会关闭您的账户。'),
('otp','Device enrollment code {code}.','设备注册验证码{code}。','Keep this code private while completing setup yourself.','请自行完成设置，并对验证码保密。','Forward the code to the salesperson to unlock a surprise bonus.','把验证码转发给销售人员才能领取意外奖金。'),
('appointment','Your clinic visit is on {day}.','您的门诊预约在{day}。','Confirm through your patient account or the clinic contact you know.','可通过就诊账户或您已知的诊所联系方式确认。','Send a cash deposit to a stranger or your medical records are published.','向陌生人交现金押金，否则公开您的病历。'),
('appointment','Repair booking {code} is confirmed.','维修预约{code}已确认。','Bring the device at the time shown in your booking.','请按预约中的时间携带设备到店。','Install our remote-control tool and open your bank app for the technician.','请安装远程控制软件，并为维修员打开银行应用。'),
('appointment','A school meeting was moved to {day}.','学校会议改至{day}。','The notice is also in the parent portal.','家长平台也可查看该通知。','The headteacher needs gift-card numbers privately to keep your child enrolled.','校长要求私下提供礼品卡号码，否则取消孩子学籍。'),
('promotion','Your subscribed bookstore alert is here.','您订阅的书店通知已送达。','View member offers if interested; STOP ends the subscription.','感兴趣可查看会员优惠，回复退订可停止接收。','Guaranteed lottery profit awaits; pay an activation fee to collect it.','保证获得彩票收益，先付激活费才能领取。'),
('promotion','Your restaurant reservation has an offer.','您的餐厅预订有相关优惠。','It is optional and your booking remains valid without it.','该优惠自愿参加，不影响原预订。','Your seat is canceled unless you disclose your full card credentials here.','不在这里提供完整银行卡凭据就取消座位。'),
('promotion','The newsletter you joined has been updated.','您加入的资讯订阅已更新。','Read it or unsubscribe; no action is required.','可阅读或退订，不需要采取其他操作。','This secret investment guarantees daily doubling if you transfer savings now.','这个秘密投资保证每日翻倍，立即转入全部积蓄即可。'),
('work','There is news about your job application.','您投递的求职申请有新进展。','An interview time is available; there is no application fee.','可预约面试时间，不收报名费。','Pay a recruitment deposit before we reveal the employer.','先付招聘押金，才能知道雇主是谁。'),
('work','Payroll reference {code} was posted.','工资流水{code}已入账。','Read the payslip in your employee account.','请在员工账户查看工资单。','Send your bank PIN to the recruiter before your wages can be released.','先把银行卡密码发给招聘人员，才能领取工资。'),
('work','Your work schedule for {day} is ready.','您在{day}的排班已发布。','Your team calendar contains the details.','详情可在团队日历查看。','Earn huge money liking posts; recharge first to unlock tasks.','给帖子点赞就能赚大钱，先充值才能解锁任务。'),
('personal','About the {amount} we discussed.','关于之前讨论的{amount}。','Let us talk face to face before deciding what to do.','先当面商量，再决定怎么处理。','I am family on a new number; transfer now and do not call to verify.','我是换了号码的家人，马上转账，不要打电话核实。'),
('travel','Booking {code} has a timetable update.','预订{code}的时刻表有调整。','Find the revised itinerary in your original travel account.','请在原旅行账户查看新行程。','A refund is locked until you send your banking one-time code.','先发送银行一次性验证码才能解锁退款。'),
('investment','Your investment statement is available.','您的投资对账单已生成。','Check returns and risks in your existing brokerage account.','请在原证券账户查看收益和风险。','Risk-free returns are guaranteed; send crypto to our personal address.','保证无风险收益，请把加密货币转到我们的私人地址。'),
]

DEV = [
('bank','A card security review has started.','银行卡安全核查已开始。','Open the bank app yourself to see the case status.','请自行打开银行应用查看处理进度。','Tell this caller your password and keep the call secret.','把密码告诉这个来电者，并对通话保密。'),
('delivery','Shipment {code} reached a pickup point.','货件{code}已到自提点。','Your original order page lists the collection address.','原订单页面列有取货地址。','A mystery fee must be wired to an unrelated person before collection.','必须先向无关人员汇入一笔不明费用才能取货。'),
('billing','A bill reminder was requested for {day}.','您申请在{day}接收账单提醒。','Review the balance through the provider you normally use.','请通过您常用的服务商核对余额。','Pay with gift-card codes here to avoid immediate arrest.','在此提供礼品卡代码缴费，否则立即逮捕。'),
('otp','Account recovery identifier {code}.','账户找回标识{code}。','Complete recovery on the screen you initiated; keep the code to yourself.','请在您发起的页面完成找回，并自行保管验证码。','Our agent needs the code so they can sign in as you.','代理需要这个验证码，以便代替您登录。'),
('appointment','Your appointment check-in opens on {day}.','您的预约签到将在{day}开放。','Use the patient portal to submit the forms.','请在就诊平台提交表格。','Give remote access to your phone and financial accounts to complete check-in.','完成签到需要向对方开放手机及金融账户的远程访问。'),
('promotion','Your requested price alert reached {amount}.','您申请的降价提醒已达到{amount}。','Buying is optional; manage alerts in your account.','是否购买由您决定，可在账户管理提醒。','Pay a prize-handling charge first and a free phone will be released.','先缴纳领奖处理费，才能发放免费手机。'),
('work','Your submitted interview documents arrived.','您提交的面试材料已收到。','The hiring team will contact you through the application portal.','招聘团队会通过求职平台联系您。','Deposit money into the recruiter wallet to qualify for the interview.','向招聘人员的钱包存钱，才能获得面试资格。'),
('personal','We need to discuss the travel budget.','我们需要讨论旅行预算。','Please call me on the number you already have.','请拨打您已保存的号码联系我。','I changed my number; buy vouchers urgently and send the secret codes.','我换号码了，请紧急购买代金券并发送其中的秘密代码。'),
]

CAL = [
('bank','Account notification {code} is ready.','账户通知{code}已就绪。','Consult the established bank portal if you want the details.','如需详情，请在已有银行平台查询。','Reveal your PIN in this reply or we seize your funds.','请在回复中泄露银行卡密码，否则没收您的资金。'),
('delivery','A parcel service update refers to {code}.','快递服务有一条关于{code}的更新。','Your courier account shows the revised pickup hours.','快递账户中可查看新的取件时间。','Send money to an unknown collector who will not provide a receipt.','请向拒绝提供收据的陌生收款人转钱。'),
('billing','A statement for {amount} has been prepared.','金额{amount}的对账单已准备好。','Use your saved provider details to verify the statement.','请用您保存的服务商信息核实账单。','Disregard your saved contact and settle with prepaid-card numbers.','不要使用已保存的联系方式，请用预付卡号码结清。'),
('otp','Verification token {code} was requested.','您申请的校验码为{code}。','Only type it into the service you are currently using.','仅在您当前使用的服务中输入。','Share it with this stranger to claim an unrelated reward.','把它交给这个陌生人，才能领取无关奖励。'),
('appointment','Your service visit has a new reference.','您预约的服务有了新编号。','The original booking account contains the updated instructions.','更新后的说明可在原预约账户查看。','The service requires your banking login and card security details by SMS.','该服务要求通过短信提交网银登录信息和银行卡安全信息。'),
('promotion','The event alerts you requested are active.','您申请的活动提醒已启用。','View the schedule or leave the list whenever you wish.','您可查看日程，也可随时退出订阅。','The event prize is guaranteed after you send an advance commission.','先支付预付佣金，就保证获得活动大奖。'),
('work','There is an update to your signed work arrangement.','您已签署的工作安排有更新。','Your employee portal has the official notice.','员工平台中有正式通知可供查看。','Your job requires a cash recharge before any tasks or wages are available.','必须先现金充值，才能获得工作任务或工资。'),
('personal','The payment discussion is still open.','关于付款的讨论尚未结束。','Check with the person through the contact you know before paying.','付款前，请通过您已知的联系方式与本人确认。','Someone claiming to be a relative demands secrecy and an instant unfamiliar transfer.','一名自称亲属的人要求保密，并立即向陌生账户转账。'),
]

TEST = [
('bank','Your card limit review is complete.','您的信用卡额度审核已结束。','The new limit is displayed after signing into your usual bank application.','登录平时使用的银行应用后可查看新额度。','A supposed manager will approve the limit once you send your account password.','所谓经理要求先发送账户密码，才批准额度。'),
('bank','A bank address change was recorded.','银行已记录地址变更。','Verify the mailing address in the account you already maintain.','请在原有账户中核实邮寄地址。','To stop the address change, disclose your full card PIN to this sender.','要阻止地址变更，请向本发送者透露完整银行卡密码。'),
('bank','Your savings interest of {amount} was credited.','存款利息{amount}已入账。','The monthly account summary shows how it was calculated.','每月账户汇总中可查看计算明细。','Unlock the interest by sending an off-platform advance tax to a personal wallet.','向私人钱包支付平台外预付税款，才能解锁利息。'),
('bank','The card you ordered is being replaced.','您申请更换的银行卡正在处理。','Activation instructions will appear in the bank account you normally use.','激活说明会出现在您平时使用的银行账户中。','The replacement agent insists on your password plus the fresh OTP you receive.','换卡代理坚持索取您的密码以及刚收到的银行验证码。'),
('delivery','Courier pickup for return {code} was booked.','退件{code}的上门取件已预约。','See the pickup window on the existing return request.','可在已有退货申请中查看取件时间段。','The courier threatens a fine unless you transfer to an unrelated private card now.','快递员威胁罚款，要求立即向无关私人银行卡转账。'),
('delivery','Package {code} arrived at a local depot.','包裹{code}已到当地仓库。','Choose a collection day using the order tracking you already have.','可使用已有的订单物流页面选择取货日期。','We demand your online-bank password to confirm the collection day.','我们要求提供网银密码，才能确认取货日期。'),
('delivery','A shopping delivery was rescheduled.','购物订单的派送时间已改。','The merchant account contains the updated delivery choices.','商家账户中列有更新后的配送选项。','The sender says your parcel will be destroyed unless an unexplained deposit is paid.','发送者声称必须支付一笔不明押金，否则销毁包裹。'),
('delivery','The missing-item refund is {amount}.','缺货退款金额为{amount}。','It will be credited to the same account used for the purchase.','款项将退回购买时使用的同一账户。','You must give this refund handler your bank PIN and security answers.','必须向本退款处理员提供银行卡密码及安全问题答案。'),
('billing','A water-service bill is due on {day}.','水费账单将在{day}到期。','Pay or ask questions through your established water account.','您可通过原有水务账户缴费或咨询。','A caller orders gift-card payments and forbids contacting the water company.','来电者要求用礼品卡缴费，并禁止联系水务公司。'),
('billing','Your insurance installment is {amount}.','本期保险分期款为{amount}。','Payment options are in your existing policy account.','已有保单账户中可查看付款方式。','The insurance agent asks you to send your banking password to prevent penalties.','保险代理要求发送银行密码，否则收取罚金。'),
('billing','A recurring charge was canceled as requested.','循环扣款已按您的申请取消。','The cancellation receipt is available in your usual subscription account.','可在平时的订阅账户查看取消回执。','To confirm cancellation, the sender requires the private code used to sign into your bank.','发送者要求提供登录银行的私密验证码，才确认取消。'),
('billing','Invoice {code} has been corrected.','发票{code}已更正。','Review the corrected document with the business you already purchased from.','请向您实际购买服务的商家核对更正后的单据。','The accountant demands an urgent cryptocurrency transfer to a secret personal address.','会计要求紧急将加密货币转到秘密私人地址。'),
('otp','Temporary access number {code}.','临时访问号码{code}。','Use it yourself on the account screen that initiated this request.','请自行在发起该申请的账户页面使用。','An unknown person requests that you forward the number so they can enter your account.','陌生人要求您转发该号码，以便其进入您的账户。'),
('otp','Your identity confirmation code is {code}.','您的身份确认码为{code}。','Do not tell anyone the code; it is for your own active session.','不要告诉他人，该码仅供您当前会话自行使用。','The sender wants both this code and your password to register as you.','发送者索取该码和您的密码，以冒用您的身份注册。'),
('otp','Account sign-on verification has started.','账户登录验证已开始。','Dismiss this notice if you are not attempting to sign in.','如您没有尝试登录，可忽略该通知。','The sender threatens arrest unless you reveal your next authentication code.','发送者威胁逮捕您，除非您透露接下来的身份验证码。'),
('otp','Security confirmation for device {code}.','设备{code}的安全确认。','Keep the verification inside the official application you opened.','请在您自行打开的官方应用内完成验证。','A stranger asks you to switch off account protection and send recovery codes.','陌生人要求您关闭账户保护，并发送恢复代码。'),
('appointment','Your vaccination appointment is on {day}.','您的疫苗接种预约在{day}。','The clinic booking system lists what to bring.','诊所预约系统中列有需要携带的材料。','A self-described clinic worker needs your card PIN by text before treatment.','一名自称诊所员工的人要求治疗前通过短信提供银行卡密码。'),
('appointment','The dentist has a cancellation opening.','牙科门诊出现了可预约空位。','Accept or decline in the patient account you already use.','您可在原就诊账户中选择接受或拒绝。','Send money to an unnamed recruiter before the appointment location is revealed.','先给不透露姓名的中间人转账，才会告知就诊地点。'),
('appointment','A vehicle inspection slot is available.','现在有车辆检查预约时段。','Confirm the slot with your original service center booking.','请通过原服务中心的预约记录确认时段。','The inspector demands remote control of your phone while you log into banking.','检查人员要求在您登录银行时远程控制您的手机。'),
('appointment','Your scheduled home service is tomorrow.','您预约的上门服务在明天。','The technician details are in the service request you submitted.','您提交的服务申请中有技术人员信息。','The sender insists you buy prepaid cards and disclose the codes before arrival.','发送者坚持要求到访前购买预付卡，并披露卡上代码。'),
('promotion','Your opted-in museum membership notice arrived.','您已选择接收的博物馆会员通知已送达。','Discount tickets are optional; leaving the mailing list is allowed.','优惠票可自愿购买，也可退出通知列表。','A lottery operator claims guaranteed winnings but requires an advance transfer first.','彩票经营者声称保证中奖，但要求先预付转账。'),
('promotion','The product availability alert you requested is ready.','您申请的商品到货提醒已就绪。','You can view stock without buying or changing account details.','您可查看库存，不必购买或更改账户信息。','The sender promises a free product only after collecting full card credentials.','发送者声称收集完整银行卡凭据后，才会寄出免费商品。'),
('promotion','Your subscribed music-event update is here.','您订阅的音乐活动更新已送达。','Manage these requested announcements in your existing membership settings.','可在原会员设置中管理您申请接收的通知。','The event allegedly doubles every investment overnight with absolutely no risk.','该活动声称任何投资都能一夜翻倍，并且绝对没有风险。'),
('promotion','A loyalty offer matches the alert you enabled.','一条会员优惠符合您启用的提醒。','Participation is your choice and the purchase account remains unchanged.','是否参加由您决定，原购买账户不会因此变更。','To refuse the offer, the sender demands your bank login and secret answers.','如不接受优惠，发送者要求提交网银登录信息及安全问题答案。'),
('work','The employer you applied to scheduled a call.','您已投递的雇主安排了通话。','Use the application record to confirm the time; no deposit is charged.','请通过求职记录确认时间，不收取押金。','A placement broker requires payment before giving you the company name.','职业中介要求先付款，才能提供公司名称。'),
('work','Your expense reimbursement is {amount}.','您的费用报销金额为{amount}。','The employee finance page shows the scheduled payment.','员工财务页面中可查看计划付款时间。','A supposed payroll clerk requires your card PIN to unlock reimbursement.','所谓工资经办人要求银行卡密码，才能解锁报销。'),
('work','The shift swap you requested was accepted.','您申请的调班已获准。','Your existing team schedule shows the revised hours.','已有团队排班表中可查看新工时。','A task agent says you must keep recharging to recover money already paid.','任务代理声称必须不断充值，才能取回已经支付的钱。'),
('work','Training for your accepted position starts on {day}.','您已接受职位的培训将在{day}开始。','Course details are in your employee portal and attendance has no fee.','课程详情在员工平台中可查，参加培训无需交费。','The recruiter guarantees a salary only after a secret cash transfer to their wallet.','招聘人员要求秘密向其钱包转入现金，之后才保证发工资。'),
('personal','We are reviewing our household payment plans.','我们正在核对家庭付款计划。','Discuss them through the phone contact we have always used.','请通过我们一直使用的电话联系方式商量。','An unfamiliar number claims a family emergency and refuses any identity check.','陌生号码声称家中有急事，并拒绝任何身份核实。'),
('travel','Your canceled train ticket has a refund update.','您取消的火车票有退款进度更新。','The original booking app displays the return to your payment account.','原订票应用中可查看退回支付账户的进度。','An unknown refund officer needs your banking passcode before releasing it.','陌生退款人员要求提供银行登录口令，才会放款。'),
('investment','A portfolio report covers the last quarter.','投资组合报告涵盖上个季度。','View both gains and losses through your established broker.','请通过原经纪商查看收益和亏损。','A contact claims every trade is profitable and tells you to borrow to invest now.','联系人声称每笔交易必赚，要求您立即借钱投资。'),
('personal','We need to verify a change to the shared account.','我们需要核实共同账户的变动。','Let us confirm it in person before authorizing any movement of money.','授权资金变动前，我们先当面确认。','The new-number sender demands instant payment and tells you not to speak to anyone else.','新号码发送者要求立即付款，并让您不要与其他任何人交谈。'),
]


def synthetic_rows():
    scenes={'train':TRAIN,'dev':DEV,'calibration':CAL,'test':TEST};result=[]
    assert [len(v) for v in scenes.values()]==[24,8,8,32]
    brands={'train':['Cedar','Maple','Willow','Elm'],'dev':['Aspen','Laurel','Hazel','Palm'],
            'calibration':['Spruce','Magnolia','Acacia','Pine'],'test':['Birch','Rowan','Alder','Holly']}
    slots={'train':8,'dev':4,'calibration':16,'test':4}
    for role,values in scenes.items():
        for index,(category,context_en,context_zh,keep_en,keep_zh,junk_en,junk_zh) in enumerate(values):
            family=f'authored-{role}-{index:02d}'
            for slot in range(slots[role]):
                args=dict(brand=brands[role][slot%4],amount=f'${17+slot*13}.40',code=str(240681+slot*719),day=f'Nov {9+slot}',
                          url=f'https://{brands[role][slot%4].lower()}.example/case/{240681+slot*719}')
                for label,action_en,action_zh in [(0,keep_en,keep_zh),(1,junk_en,junk_zh)]:
                    contents={'en':context_en+' '+action_en,'zh':context_zh+action_zh,
                              'mixed':context_zh+' '+action_en if index%2==0 else context_en+' '+action_zh}
                    for language,text in contents.items():
                        text=(f'{args["brand"]}: '+text+' Ref {code}. {url}').format(**args)
                        result.append(dict(id=f'{family}:{slot}:{language}:{label}',family_id=family,template_id=family,
                            label=label,text=text,language=language,panel='new-'+language,role=role,source='original-authored-20261105',
                            message_type=category,synthetic=True,actual_received_sms=False,pair_id=f'{family}:{slot}',
                            provenance='Original fictional context/actions; mixed is bilingual composition; slots correlated'))
    return result

